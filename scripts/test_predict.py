import json
import requests

API = "http://localhost:8000"

payload = {
    "records": [
        {"Store": 1,  "DayOfWeek": 5, "Date": "2015-09-17", "Open": 1, "Promo": 1, "StateHoliday": "0", "SchoolHoliday": 0},
        {"Store": 2,  "DayOfWeek": 3, "Date": "2015-09-15", "Open": 1, "Promo": 0, "StateHoliday": "0", "SchoolHoliday": 1},
        {"Store": 5,  "DayOfWeek": 1, "Date": "2015-09-14", "Open": 1, "Promo": 1, "StateHoliday": "0", "SchoolHoliday": 0},
        {"Store": 10, "DayOfWeek": 6, "Date": "2015-09-19", "Open": 1, "Promo": 0, "StateHoliday": "a", "SchoolHoliday": 0},
    ]
}

print("=== Health Check ===")
health = requests.get(f"{API}/health").json()
print(health)

print("\n=== Predict ===")
resp = requests.post(f"{API}/predict", json=payload)
data = resp.json()
print(f"Status: {resp.status_code} | Count: {data['count']}")
print()
for rec, pred in zip(payload["records"], data["predictions"]):
    print(f"  Store {rec['Store']:>4} | {rec['Date']} | Promo={rec['Promo']} -> EUR {pred:>10,.2f}")
