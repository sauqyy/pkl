import urllib.request
import json
import re
import os

ACCESS_TOKEN = "eyJraWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBcHBEeW5hbWljcyIsImF1ZCI6IkFwcERfQVBJcyIsImp0aSI6InRzVTRTazEtazZGOURrXzlTWGVvS3ciLCJzdWIiOiJtYWdhbmciLCJpZFR5cGUiOiJBUElfQ0xJRU5UIiwiaWQiOiI4MTFkYjUzMi1iYWNjLTQ2YmYtOGI5NC01ZjU2OTdkOTMwMTAiLCJhY2N0SWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJ0bnRJZCI6IjVmYzkxOTg4LWExZjgtNDhlMS04YjIwLTlmZjA4NjRkYmFmNiIsImFjY3ROYW1lIjoicHRmd2RpbmRvbmVzaWEtcHJvZCIsInRlbmFudE5hbWUiOiIiLCJmbW1UbnRJZCI6bnVsbCwiYWNjdFBlcm0iOltdLCJyb2xlSWRzIjpbXSwiaWF0IjoxNzY4MjEwNTg2LCJuYmYiOjE3NjgyMTA0NjYsImV4cCI6MTc3MDgwMjU4NiwidG9rZW5UeXBlIjoiQUNDRVNTIn0.3bq0IN3UL5_ZoFwk1mcKYfs776miYviFN6GmTV1XRa8"

# URL for /smart-integration/users - Error Metric
base_url = "https://ptfwdindonesia-prod.saas.appdynamics.com/controller/rest/applications/Smart2/metric-data?metric-path=Business%20Transaction%20Performance%7CBusiness%20Transactions%7Cintegration-service%7C%2Fsmart-integration%2Fusers%7CErrors%20per%20Minute&time-range-type=BEFORE_NOW&duration-in-mins=43200"

def fetch(label, duration, allow_rollup):
    print(f"\n--- Testing {label} (Duration={duration}, Rollup={allow_rollup}) ---")
    url = base_url
    if duration:
        if 'duration-in-mins=' in url:
            url = re.sub(r'duration-in-mins=\d+', f'duration-in-mins={duration}', url)
    
    if not allow_rollup:
        url += '&rollup=false'
    else:
        url += '&rollup=true' # Explicitly enable for test
        
    url += '&output=JSON'
    
    print(f"URL: {url}")
    
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {ACCESS_TOKEN}")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            
            # Flatten to find first value
            values = []
            if isinstance(data, list):
                for item in data:
                    if 'metricValues' in item:
                        values.extend(item['metricValues'])
            
            print(f"Total Items: {len(values)}")
            if len(values) > 0:
                print(f"First Item Keys: {values[0].keys()}")
                print(f"First Item Content: {values[0]}")
                
                # Check sum vs value
                sample = values[0]
                val = sample.get('value', 0)
                sum_val = sample.get('sum', -1)
                count = sample.get('count', -1)
                min_val = sample.get('min', -1)
                max_val = sample.get('max', -1)
                
                print(f"Value: {val}, Sum: {sum_val}, Count: {count}, Min: {min_val}, Max: {max_val}")
            
    except Exception as e:
        print(f"Error: {e}")

# Test "Last 6 Months" (approx 260,000 mins)
fetch("Last 6 Months", 260000, True)
