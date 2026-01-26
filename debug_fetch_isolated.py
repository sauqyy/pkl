import urllib.request
import json
import re
import time
import os

ACCESS_TOKEN = "eyJraWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBcHBEeW5hbWljcyIsImF1ZCI6IkFwcERfQVBJcyIsImp0aSI6InRzVTRTazEtazZGOURrXzlTWGVvS3ciLCJzdWIiOiJtYWdhbmciLCJpZFR5cGUiOiJBUElfQ0xJRU5UIiwiaWQiOiI4MTFkYjUzMi1iYWNjLTQ2YmYtOGI5NC01ZjU2OTdkOTMwMTAiLCJhY2N0SWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJ0bnRJZCI6IjVmYzkxOTg4LWExZjgtNDhlMS04YjIwLTlmZjA4NjRkYmFmNiIsImFjY3ROYW1lIjoicHRmd2RpbmRvbmVzaWEtcHJvZCIsInRlbmFudE5hbWUiOiIiLCJmbW1UbnRJZCI6bnVsbCwiYWNjdFBlcm0iOltdLCJyb2xlSWRzIjpbXSwiaWF0IjoxNzY4MjEwNTg2LCJuYmYiOjE3NjgyMTA0NjYsImV4cCI6MTc3MDgwMjU4NiwidG9rZW5UeXBlIjoiQUNDRVNTIn0.3bq0IN3UL5_ZoFwk1mcKYfs776miYviFN6GmTV1XRa8"

# URL from CSV for /smart-integration/users - Error Metric
base_url = "https://ptfwdindonesia-prod.saas.appdynamics.com/controller/rest/applications/Smart2/metric-data?metric-path=Business%20Transaction%20Performance%7CBusiness%20Transactions%7Cintegration-service%7C%2Fsmart-integration%2Fusers%7CErrors%20per%20Minute&time-range-type=BEFORE_NOW&duration-in-mins=43200"

def fetch(url, duration_override):
    if duration_override:
        if 'duration-in-mins=' in url:
            url = re.sub(r'duration-in-mins=\d+', f'duration-in-mins={duration_override}', url)
    
    url += '&output=JSON&rollup=false'
    
    print(f"Fetching: {url}")
    
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {ACCESS_TOKEN}")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            print(f"Response Status: {response.getcode()}")
            
            # Analyze data
            count = 0
            if data and len(data) > 0 and 'metricValues' in data[0]:
                values = data[0]['metricValues']
                count = len(values)
                print(f"Items returned: {count}")
                if count > 0:
                    print(f"Last Item: {values[-1]}")
                    # Print timestamp of last item
                    ts = values[-1]['startTimeInMillis']
                    import datetime
                    dt = datetime.datetime.fromtimestamp(ts/1000)
                    print(f"Last Timestamp: {dt}")
            else:
                 print("No metricValues found.")
                 
    except Exception as e:
        print(f"Error: {e}")

print("--- Testing duration=15 ---")
fetch(base_url, 15)

print("\n--- Testing duration=60 ---")
fetch(base_url, 60)
