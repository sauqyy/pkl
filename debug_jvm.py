import urllib.request
import urllib.parse
import json

# Config from app.py
ACCESS_TOKEN = "eyJraWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBcHBEeW5hbWljcyIsImF1ZCI6IkFwcERfQVBJcyIsImp0aSI6InRzVTRTazEtazZGOURrXzlTWGVvS3ciLCJzdWIiOiJtYWdhbmciLCJpZFR5cGUiOiJBUElfQ0xJRU5UIiwiaWQiOiI4MTFkYjUzMi1iYWNjLTQ2YmYtOGI5NC01ZjU2OTdkOTMwMTAiLCJhY2N0SWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJ0bnRJZCI6IjVmYzkxOTg4LWExZjgtNDhlMS04YjIwLTlmZjA4NjRkYmFmNiIsImFjY3ROYW1lIjoicHRmd2RpbmRvbmVzaWEtcHJvZCIsInRlbmFudE5hbWUiOiIiLCJmbW1UbnRJZCI6bnVsbCwiYWNjdFBlcm0iOltdLCJyb2xlSWRzIjpbXSwiaWF0IjoxNzY4MjEwNTg2LCJuYmYiOjE3NjgyMTA0NjYsImV4cCI6MTc3MDgwMjU4NiwidG9rZW5UeXBlIjoiQUNDRVNTIn0.3bq0IN3UL5_ZoFwk1mcKYfs776miYviFN6GmTV1XRa8"
BASE_URL = "https://ptfwdindonesia-prod.saas.appdynamics.com"
APP_NAME = "SmartNano"

# Metric from user request (decoded first to be safe, then let code encode it)
# "Application Infrastructure Performance|dynamic-letter-deployment|Agent|App|Availability"
METRIC_PATH = "Application Infrastructure Performance|dynamic-letter-deployment|Agent|App|Availability"

def test_fetch(metric_path):
    print(f"Testing metric: {metric_path}")
    encoded_metric = urllib.parse.quote(metric_path)
    url = (
        f"{BASE_URL}/controller/rest/applications/{APP_NAME}/metric-data?"
        f"metric-path={encoded_metric}&"
        f"time-range-type=BEFORE_NOW&"
        f"duration-in-mins=60&"
        f"output=JSON&"
        f"rollup=false"
    )
    print(f"URL: {url}")
    
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {ACCESS_TOKEN}")
    try:
        with urllib.request.urlopen(req) as response:
            data = response.read().decode()
            print("Response Code:", response.getcode())
            print("Response Data Snippet:", data[:500])
            parsed = json.loads(data)
            if isinstance(parsed, list) and len(parsed) > 0:
                vals = parsed[0].get('metricValues', [])
                print(f"Found {len(vals)} values.")
            else:
                print("Empty list returned (No data found for this path/time).")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        print(e.read().decode())
    except Exception as e:
        print(f"Error: {e}")

test_fetch(METRIC_PATH)

print("-" * 50)
# Test Heap
HEAP = "Application Infrastructure Performance|dynamic-letter-deployment|Individual Nodes|dynamic-letter-deployment--18|JVM|Memory|Heap|Used %"
test_fetch(HEAP)
