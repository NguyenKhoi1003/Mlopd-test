import argparse
import json
import os
import platform

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import yaml
from catboost import CatBoostRegressor
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import logging

# -----------------------------
# 1. Cấu hình Logging & Metrics
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def rmspe(y_true, y_pred):
    """Tính toán RMSPE cho bài toán Rossmann"""
    return np.sqrt(np.mean(((y_true - y_pred) / y_true)**2))

# -----------------------------
# 2. Khởi tạo mô hình (Factory Pattern)
# -----------------------------
def get_model_instance(name, params):
    model_map = {
        'LinearRegression': LinearRegression,
        'XGBoost': xgb.XGBRegressor,
        'LightGBM': lgb.LGBMRegressor,
        'CatBoost': CatBoostRegressor
    }
    if name not in model_map:
        raise ValueError(f"Mô hình không hỗ trợ: {name}")
    return model_map[name](**params)

# -----------------------------
# 3. Pipeline Function (for run_pipeline.py)
# -----------------------------
def train_pipeline(config):
    """
    Main training pipeline function that accepts config dict.
    Handles both raw data (with store merge + feature engineering)
    and pre-processed data (Sales_log already present).
    """
    from src.rossmann_mlops.features import run_feature_engineering
    from src.rossmann_mlops.processing import merge_data, preprocess_data

    # Extract config values with defaults
    training_cfg = config.get('training', {})
    paths_cfg = config.get('paths', {})

    model_name = 'XGBoost'
    model_params = {
        'n_estimators': training_cfg.get('n_estimators', 300),
        'random_state': training_cfg.get('random_state', 42),
    }

    train_data_path = paths_cfg.get('train_data', 'data/processed/train_final.csv')
    store_data_path = paths_cfg.get('store_data', 'data/raw/store.csv')
    model_save_path = paths_cfg.get('model_file', 'artifacts/models/rossmann_model.joblib')
    metrics_save_path = paths_cfg.get('metrics_file', 'artifacts/metrics/metrics.json')

    # Setup MLflow — use local file tracking by default (no server required)
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', '')
    mlflow.set_tracking_uri(mlflow_uri if mlflow_uri else 'mlruns')
    mlflow.set_experiment("Rossmann_Sales_Pipeline")

    # Load data
    logger.info(f"Đang tải dữ liệu từ: {train_data_path}")
    try:
        raw_train = pd.read_csv(train_data_path, dtype={"StateHoliday": str})
        store_df  = pd.read_csv(store_data_path)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return {'status': 'error', 'message': str(exc)}

    # If data is already pre-processed (has Sales_log), skip feature engineering
    if 'Sales_log' in raw_train.columns:
        train_featured = raw_train.copy()
    else:
        # Merge store data, preprocess, build features
        train_merged, _ = merge_data(raw_train, raw_train.head(0), store_df)
        train_processed, _ = preprocess_data(train_merged, train_merged.head(0))
        train_featured, _ = run_feature_engineering(train_processed, train_processed.head(0))

    # Time-based split
    if 'Year' in train_featured.columns and 'WeekOfYear' in train_featured.columns:
        val_condition = (train_featured['Year'] == 2015) & (train_featured['WeekOfYear'] >= 26)
    else:
        val_condition = train_featured.index >= int(len(train_featured) * 0.8)

    train_df = train_featured[~val_condition].copy()
    val_df   = train_featured[val_condition].copy()

    drop_cols = ['Sales', 'Sales_log', 'Customers']
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train = train_df['Sales_log'] if 'Sales_log' in train_df.columns else train_df['Sales']
    X_val   = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])
    y_val   = val_df['Sales_log'] if 'Sales_log' in val_df.columns else val_df.get('Sales', pd.Series(dtype=float))

    model = get_model_instance(model_name, model_params)

    try:
        with mlflow.start_run(run_name="pipeline_training"):
            logger.info(f"🚀 Đang huấn luyện: {model_name}")
            model.fit(X_train, y_train)

            if len(X_val) > 0 and len(y_val) > 0:
                y_pred     = model.predict(X_val)
                log_scale  = 'Sales_log' in train_featured.columns
                y_true_s   = np.expm1(y_val.values)  if log_scale else y_val.values
                y_pred_s   = np.expm1(y_pred)         if log_scale else y_pred
                rmse_val   = float(np.sqrt(mean_squared_error(y_true_s, y_pred_s)))
                mae_val    = float(mean_absolute_error(y_true_s, y_pred_s))
                r2_val     = float(r2_score(y_true_s, y_pred_s))
                rmspe_val  = float(rmspe(y_true_s, y_pred_s))
            else:
                rmse_val = mae_val = r2_val = rmspe_val = 0.0

            metrics = {'rmse': rmse_val, 'mae': mae_val, 'r2': r2_val, 'rmspe': rmspe_val}

            mlflow.log_params(model_params)
            mlflow.log_metrics(metrics)

            # Save model
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            joblib.dump(model, model_save_path)

            # Save metrics file
            os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
            with open(metrics_save_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"✅ RMSE: {rmse_val:.2f} | MAE: {mae_val:.2f} | R2: {r2_val:.4f}")
            return {
                'status': 'success',
                'model_path': model_save_path,
                'metrics_path': metrics_save_path,
                'metrics': metrics,
            }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {'status': 'error', 'message': str(e)}

# -----------------------------
# 4. Luồng xử lý chính
# -----------------------------
def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_cfg = config['best_model'] # Lưu ý: File config Đa lưu ở cell trước có key 'best_model'

    # Setup MLflow
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("Rossmann_Final_Training")

    # Load data
    logger.info(f"Đang tải dữ liệu từ: {args.data}")
    df = pd.read_csv(args.data)
    
    # Time-based split (Logic đặc thù cho Rossmann của Đa)
    val_condition = (df['Year'] == 2015) & (df['WeekOfYear'] >= 26)
    train_df = df[~val_condition].copy()
    val_df = df[val_condition].copy()

    # Tách feature/target (Tránh Leakage)
    drop_cols = ['Sales', 'Sales_log', 'Customers']
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df['Sales_log']
    
    X_val = val_df.drop(columns=drop_cols, errors='ignore')
    y_val_orig = np.exp(val_df['Sales_log']) # Giá trị thực tế để tính RMSPE

    # Get model
    model = get_model_instance(model_cfg['name'], model_cfg['params'])

    # Start MLflow run
    with mlflow.start_run(run_name="final_production_training"):
        logger.info(f"🚀 Đang huấn luyện: {model_cfg['name']}")
        model.fit(X_train, y_train)
        
        # Dự đoán & Tính RMSPE
        y_pred_log = model.predict(X_val)
        y_pred_orig = np.exp(y_pred_log)
        val_rmspe = float(rmspe(y_val_orig, y_pred_orig))

        # Log params và metrics (Cập nhật sang RMSPE)
        mlflow.log_params(model_cfg['params'])
        mlflow.log_metric('val_rmspe', val_rmspe)

        # Đăng ký Model vào Registry (Theo chuẩn Skeleton)
        mlflow.sklearn.log_model(model, "production_model")
        reg_model_name = "Rossmann_Sales_Model" # Tên định danh chung của nhóm
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/production_model"

        logger.info("Đang đăng ký mô hình vào MLflow Registry...")
        client = MlflowClient()
        try:
            client.create_registered_model(reg_model_name)
        except:
            pass # Đã tồn tại

        model_version = client.create_model_version(
            name=reg_model_name,
            source=model_uri,
            run_id=mlflow.active_run().info.run_id
        )

        # Chuyển sang Staging
        client.transition_model_version_stage(
            name=reg_model_name,
            version=model_version.version,
            stage="Staging"
        )

        # Cập nhật Description & Tags chi tiết
        description = (
            f"Mô hình dự báo doanh số Rossmann.\n"
            f"Thuật toán: {model_cfg['name']}\n"
            f"RMSPE trên tập 6 tuần cuối: {val_rmspe:.4f}"
        )
        client.update_registered_model(name=reg_model_name, description=description)
        
        # Thêm các thẻ phụ (Tags) giúp tra cứu nhanh
        tags = {
            "algorithm": model_cfg['name'],
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "training_date": pd.Timestamp.now().strftime("%Y-%m-%d")
        }
        for k, v in tags.items():
            client.set_registered_model_tag(reg_model_name, k, v)

        # Lưu model cục bộ (pkl)
        os.makedirs(args.models_dir, exist_ok=True)
        save_path = os.path.join(args.models_dir, "rossmann_best_model.pkl")
        joblib.dump(model, save_path)
        
        logger.info(f"✅ Hoàn tất! Model saved: {save_path} | RMSPE: {val_rmspe:.4f}")

# -----------------------------
# 4. Cấu hình Argument Parser
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn file model_config.yaml")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn file dữ liệu CSV đã featured")
    parser.add_argument("--models-dir", type=str, default="../models/trained", help="Thư mục lưu model")
    parser.add_argument("--mlflow-uri", type=str, default="http://127.0.0.1:5000", help="MLflow Server URI")
    
    args = parser.parse_args()
    main(args)