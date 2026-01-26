import urllib.request
import json
import re
import datetime
import pandas as pd
from collections import Counter

ACCESS_TOKEN = "eyJraWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBcHBEeW5hbWljcyIsImF1ZCI6IkFwcERfQVBJcyIsImp0aSI6InRzVTRTazEtazZGOURrXzlTWGVvS3ciLCJzdWIiOiJtYWdhbmciLCJpZFR5cGUiOiJBUElfQ0xJRU5UIiwiaWQiOiI4MTFkYjUzMi1iYWNjLTQ2YmYtOGI5NC01ZjU2OTdkOTMwMTAiLCJhY2N0SWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJ0bnRJZCI6IjVmYzkxOTg4LWExZjgtNDhlMS04YjIwLTlmZjA4NjRkYmFmNiIsImFjY3ROYW1lIjoicHRmd2RpbmRvbmVzaWEtcHJvZCIsInRlbmFudE5hbWUiOiIiLCJmbW1UbnRJZCI6bnVsbCwiYWNjdFBlcm0iOltdLCJyb2xlSWRzIjpbXSwiaWF0IjoxNzY4MjEwNTg2LCJuYmYiOjE3NjgyMTA0NjYsImV4cCI6MTc3MDgwMjU4NiwidG9rZW5UeXBlIjoiQUNDRVNTIn0.3bq0IN3UL5_ZoFwk1mcKYfs776miYviFN6GmTV1XRa8"

# URL for /smart-integration/users - Error Metric
base_url = "https://ptfwdindonesia-prod.saas.appdynamics.com/controller/rest/applications/Smart2/metric-data?metric-path=Business%20Transaction%20Performance%7CBusiness%20Transactions%7Cintegration-service%7C%2Fsmart-integration%2Fusers%7CErrors%20per%20Minute&time-range-type=BEFORE_NOW"

# Manually implement fetch logic mirroring app.py
def fetch_and_analyze(label, duration, allow_rollup):
    print(f"\n--- Testing {label} (Duration={duration}, Rollup={allow_rollup}) ---")
    url = base_url + f"&duration-in-mins={duration}&output=JSON"
    
    if not allow_rollup:
        url += '&rollup=false'
    # if allow_rollup, we default to whatever AppD does (usually rollup=true)
    
    print(f"URL: {url}")
    
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {ACCESS_TOKEN}")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            
            # Flatten like app.py
            all_values = []
            if isinstance(data, list):
                for item in data:
                    if 'metricValues' in item:
                        all_values.extend(item['metricValues'])
            
            print(f"Total Points: {len(all_values)}")
            
            if not all_values:
                print("No metricValues found.")
                return

            # Analyze Timestamps
            df = pd.DataFrame(all_values)
            df['dt'] = pd.to_datetime(df['startTimeInMillis'], unit='ms') + pd.Timedelta(hours=7)
            df['date'] = df['dt'].dt.date
            df['day_name'] = df['dt'].dt.day_name()
            
            # Check unique dates
            dates = df['date'].unique()
            print(f"Unique Dates Found: {len(dates)}")
            print(f"Dates: {sorted(dates)}")
            
            # Check Day Names spread
            day_counts = df['day_name'].value_counts()
            print("Day Distribution:")
            print(day_counts)
            
            # Check first/last timestamp
            print(f"Min Time: {df['dt'].min()}")
            print(f"Max Time: {df['dt'].max()}")
            
    except Exception as e:
        print(f"Error: {e}")

# Test 7 Days (10080 mins) - Expect 7 unique dates
fetch_and_analyze("Last 7 Days", 10080, True)

# Test 30 Days (43200 mins)
fetch_and_analyze("Last 30 Days", 43200, True)
