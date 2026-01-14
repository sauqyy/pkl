import time
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import os

# --- Configuration ---
BASE_URL = "https://ptfwdindonesia-prod.saas.appdynamics.com"
APP_NAME = "Smart2" 
# Metric Paths
METRIC_SLOW = "Business Transaction Performance|Business Transactions|integration-service|/smart-integration/users|Number of Slow Calls"
METRIC_VERY_SLOW = "Business Transaction Performance|Business Transactions|integration-service|/smart-integration/users|Number of Very Slow Calls"

ACCESS_TOKEN = "eyJraWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBcHBEeW5hbWljcyIsImF1ZCI6IkFwcERfQVBJcyIsImp0aSI6InRzVTRTazEtazZGOURrXzlTWGVvS3ciLCJzdWIiOiJtYWdhbmciLCJpZFR5cGUiOiJBUElfQ0xJRU5UIiwiaWQiOiI4MTFkYjUzMi1iYWNjLTQ2YmYtOGI5NC01ZjU2OTdkOTMwMTAiLCJhY2N0SWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJ0bnRJZCI6IjVmYzkxOTg4LWExZjgtNDhlMS04YjIwLTlmZjA4NjRkYmFmNiIsImFjY3ROYW1lIjoicHRmd2RpbmRvbmVzaWEtcHJvZCIsInRlbmFudE5hbWUiOiIiLCJmbW1UbnRJZCI6bnVsbCwiYWNjdFBlcm0iOltdLCJyb2xlSWRzIjpbXSwiaWF0IjoxNzY4MjEwNTg2LCJuYmYiOjE3NjgyMTA0NjYsImV4cCI6MTc3MDgwMjU4NiwidG9rZW5UeXBlIjoiQUNDRVNTIn0.3bq0IN3UL5_ZoFwk1mcKYfs776miYviFN6GmTV1XRa8"

FILE_SLOW = "slow_calls_data.json"
FILE_VERY_SLOW = "very_slow_calls_data.json"

def get_metric_data(start_dt, end_dt, metric_path):
    """
    Fetches metric data for a specific time range.
    """
    start_epoch = int(start_dt.timestamp() * 1000)
    end_epoch = int(end_dt.timestamp() * 1000)
    
    encoded_metric = urllib.parse.quote(metric_path)
    
    url = (
        f"{BASE_URL}/controller/rest/applications/{APP_NAME}/metric-data?"
        f"metric-path={encoded_metric}&"
        f"time-range-type=BETWEEN_TIMES&"
        f"start-time={start_epoch}&"
        f"end-time={end_epoch}&"
        f"output=JSON&"
        f"rollup=false" 
    )
    
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {ACCESS_TOKEN}")
    
    try:
        # print(f"Fetching data from {start_dt} to {end_dt}...")
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def fetch_and_save(metric_path, output_file, duration_days=365):
    print(f"\n--- Processing {output_file} ---")
    now = datetime.now()
    start_date = now - timedelta(days=duration_days) 
    
    all_data = []
    
    current_start = start_date
    chunk_size = timedelta(days=7)
    
    while current_start < now:
        current_end = min(current_start + chunk_size, now)
        
        print(f"Fetching {current_start.date()} to {current_end.date()}...")
        chunk_data = get_metric_data(current_start, current_end, metric_path)
        
        if chunk_data and isinstance(chunk_data, list):
            for item in chunk_data:
                if 'metricValues' in item:
                    all_data.extend(item['metricValues'])
        
        current_start = current_end
        # time.sleep(0.5) 
        
    print(f"Total data points: {len(all_data)}")
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved to {output_file}")

def main():
    # Fetch Slow Calls
    fetch_and_save(METRIC_SLOW, FILE_SLOW)
    
    # Fetch Very Slow Calls
    fetch_and_save(METRIC_VERY_SLOW, FILE_VERY_SLOW)

if __name__ == "__main__":
    main()
