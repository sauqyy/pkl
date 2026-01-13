import time
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import os

# --- Configuration ---
BASE_URL = "https://ptfwdindonesia-prod.saas.appdynamics.com"
APP_NAME = "SmartNano"
METRIC_PATH = "Overall Application Performance|Calls per Minute"
# Using the same token
ACCESS_TOKEN = "eyJraWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBcHBEeW5hbWljcyIsImF1ZCI6IkFwcERfQVBJcyIsImp0aSI6InRzVTRTazEtazZGOURrXzlTWGVvS3ciLCJzdWIiOiJtYWdhbmciLCJpZFR5cGUiOiJBUElfQ0xJRU5UIiwiaWQiOiI4MTFkYjUzMi1iYWNjLTQ2YmYtOGI5NC01ZjU2OTdkOTMwMTAiLCJhY2N0SWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJ0bnRJZCI6IjVmYzkxOTg4LWExZjgtNDhlMS04YjIwLTlmZjA4NjRkYmFmNiIsImFjY3ROYW1lIjoicHRmd2RpbmRvbmVzaWEtcHJvZCIsInRlbmFudE5hbWUiOiIiLCJmbW1UbnRJZCI6bnVsbCwiYWNjdFBlcm0iOltdLCJyb2xlSWRzIjpbXSwiaWF0IjoxNzY4MjEwNTg2LCJuYmYiOjE3NjgyMTA0NjYsImV4cCI6MTc3MDgwMjU4NiwidG9rZW5UeXBlIjoiQUNDRVNTIn0.3bq0IN3UL5_ZoFwk1mcKYfs776miYviFN6GmTV1XRa8"

OUTPUT_FILE = "load_data.json"

def get_metric_data(start_dt, end_dt):
    """
    Fetches metric data for a specific time range.
    """
    start_epoch = int(start_dt.timestamp() * 1000)
    end_epoch = int(end_dt.timestamp() * 1000)
    
    encoded_metric = urllib.parse.quote(METRIC_PATH)
    
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
        print(f"Fetching data from {start_dt} to {end_dt}...")
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def main():
    # Calculate "Lifetime" (approx 3 years back)
    now = datetime.now()
    long_ago = now - timedelta(days=1095) # 3 Years
    
    all_data = []
    
    # Fetch in 30-day chunks
    current_start = long_ago
    chunk_size = timedelta(days=30)
    
    while current_start < now:
        current_end = min(current_start + chunk_size, now)
        
        chunk_data = get_metric_data(current_start, current_end)
        
        # Extract values
        if chunk_data and isinstance(chunk_data, list):
            for item in chunk_data:
                if 'metricValues' in item:
                    all_data.extend(item['metricValues'])
        
        current_start = current_end
        time.sleep(1) # Be polite
        
    # Save to file
    print(f"Total data points collected: {len(all_data)}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
