import urllib.request
import json
import time

# Credentials provided by user
ACCESS_TOKEN = "eyJraWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBcHBEeW5hbWljcyIsImF1ZCI6IkFwcERfQVBJcyIsImp0aSI6InRzVTRTazEtazZGOURrXzlTWGVvS3ciLCJzdWIiOiJtYWdhbmciLCJpZFR5cGUiOiJBUElfQ0xJRU5UIiwiaWQiOiI4MTFkYjUzMi1iYWNjLTQ2YmYtOGI5NC01ZjU2OTdkOTMwMTAiLCJhY2N0SWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJ0bnRJZCI6IjVmYzkxOTg4LWExZjgtNDhlMS04YjIwLTlmZjA4NjRkYmFmNiIsImFjY3ROYW1lIjoicHRmd2RpbmRvbmVzaWEtcHJvZCIsInRlbmFudE5hbWUiOiIiLCJmbW1UbnRJZCI6bnVsbCwiYWNjdFBlcm0iOltdLCJyb2xlSWRzIjpbXSwiaWF0IjoxNzY4MjEwNTg2LCJuYmYiOjE3NjgyMTA0NjYsImV4cCI6MTc3MDgwMjU4NiwidG9rZW5UeXBlIjoiQUNDRVNTIn0.3bq0IN3UL5_ZoFwk1mcKYfs776miYviFN6GmTV1XRa8"

# URL for testing (Last 15 mins)
BASE_URL = "https://ptfwdindonesia-prod.saas.appdynamics.com"
APP_NAME = "SmartNano"
METRIC_PATH = "Overall Application Performance|Average Response Time (ms)"
encoded_metric = urllib.parse.quote(METRIC_PATH)

url = (
    f"{BASE_URL}/controller/rest/applications/{APP_NAME}/metric-data?"
    f"metric-path={encoded_metric}&"
    f"time-range-type=BEFORE_NOW&"
    f"duration-in-mins=15&"
    f"output=JSON&"
    f"rollup=false"
)

print(f"Connecting to: {url}")

req = urllib.request.Request(url)
req.add_header("Authorization", f"Bearer {ACCESS_TOKEN}")

try:
    with urllib.request.urlopen(req) as response:
        print("SUCCESS! Connection established.")
        data = json.loads(response.read().decode())
        print(json.dumps(data, indent=2))
        
        # Save to file to verify we can update the dashboard data
        with open('metric-data.json', 'w') as f:
            json.dump(data, f, indent=2)
            print("\nSaved live data to metric-data.json")
            
except Exception as e:
    print(f"FAILED: {e}")
    if hasattr(e, 'headers'):
        print("Headers:", e.headers)
    if hasattr(e, 'read'):
        print("Body:", e.read().decode())
