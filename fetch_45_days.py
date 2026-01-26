import time
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import os

# --- Configuration ---
BASE_URL = "https://ptfwdindonesia-prod.saas.appdynamics.com"
APP_NAME = "SmartNano"

# Metric Paths
METRIC_LOAD = "Overall Application Performance|Calls per Minute"
METRIC_RESPONSE = "Overall Application Performance|Average Response Time (ms)"
METRIC_ERROR = "Overall Application Performance|Errors per Minute"
# Special path for Slow Calls (Integration Service BT)
METRIC_SLOW = "Business Transaction Performance|Business Transactions|integration-service|/smart-integration/users|Number of Slow Calls"
APP_NAME_SLOW = "Smart2" # Note: Slow calls used "Smart2" in fetch_slow_calls.py

ACCESS_TOKEN = "eyJraWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBcHBEeW5hbWljcyIsImF1ZCI6IkFwcERfQVBJcyIsImp0aSI6InRzVTRTazEtazZGOURrXzlTWGVvS3ciLCJzdWIiOiJtYWdhbmciLCJpZFR5cGUiOiJBUElfQ0xJRU5UIiwiaWQiOiI4MTFkYjUzMi1iYWNjLTQ2YmYtOGI5NC01ZjU2OTdkOTMwMTAiLCJhY2N0SWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJ0bnRJZCI6IjVmYzkxOTg4LWExZjgtNDhlMS04YjIwLTlmZjA4NjRkYmFmNiIsImFjY3ROYW1lIjoicHRmd2RpbmRvbmVzaWEtcHJvZCIsInRlbmFudE5hbWUiOiIiLCJmbW1UbnRJZCI6bnVsbCwiYWNjdFBlcm0iOltdLCJyb2xlSWRzIjpbXSwiaWF0IjoxNzY4MjEwNTg2LCJuYmYiOjE3NjgyMTA0NjYsImV4cCI6MTc3MDgwMjU4NiwidG9rZW5UeXBlIjoiQUNDRVNTIn0.3bq0IN3UL5_ZoFwk1mcKYfs776miYviFN6GmTV1XRa8"

OUTPUT_DIR = "model"

def get_metric_data(start_dt, end_dt, metric_path, app_name=APP_NAME):
    """
    Fetches metric data for a specific time range.
    """
    start_epoch = int(start_dt.timestamp() * 1000)
    end_epoch = int(end_dt.timestamp() * 1000)
    
    encoded_metric = urllib.parse.quote(metric_path)
    
    url = (
        f"{BASE_URL}/controller/rest/applications/{app_name}/metric-data?"
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
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching data for {metric_path}: {e}")
        return []

def fetch_and_save(metric_path, output_filename, duration_days=45, app_name=APP_NAME):
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    print(f"\n--- Processing {output_filename} ---")
    now = datetime.now()
    start_date = now - timedelta(days=duration_days) 
    
    all_data = []
    
    current_start = start_date
    chunk_size = timedelta(days=7) # Fetch in 7-day chunks to avoid timeouts
    
    print(f"Fetching {duration_days} days of data for: {metric_path}")
    
    while current_start < now:
        current_end = min(current_start + chunk_size, now)
        
        print(f"  Fetching {current_start.date()} to {current_end.date()}...")
        chunk_data = get_metric_data(current_start, current_end, metric_path, app_name)
        
        if chunk_data and isinstance(chunk_data, list):
            for item in chunk_data:
                if 'metricValues' in item:
                    values = item['metricValues']
                    if values:
                        print(f"    Found {len(values)} points")
                        all_data.extend(values)
                    else:
                        print(f"    No values in chunk")
        
        current_start = current_end
        time.sleep(0.5) 
        
    print(f"Total data points collected: {len(all_data)}")
    
    # Sort by timestamp just in case
    all_data.sort(key=lambda x: x['startTimeInMillis'])
    
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved to {output_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Fetch Load
    fetch_and_save(METRIC_LOAD, "data_load.json")
    
    # 2. Fetch Response
    fetch_and_save(METRIC_RESPONSE, "data_response.json")
    
    # 3. Fetch Error
    fetch_and_save(METRIC_ERROR, "data_error.json")
    
    # 4. Fetch Slow Calls (Note: Uses Smart2 app name as per original script)
    fetch_and_save(METRIC_SLOW, "data_slow.json", app_name=APP_NAME_SLOW)

    print("\n" + "="*50)
    print("ALL DONE! Files saved in 'model/' directory.")
    print("="*50)

if __name__ == "__main__":
    main()
