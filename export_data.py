import json
import csv
import os
import urllib.request
import urllib.parse
import hashlib
import time
import pandas as pd
import re
from datetime import datetime

# --- Configuration (Copied from app.py to avoid import side-effects) ---
ACCESS_TOKEN = "eyJraWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBcHBEeW5hbWljcyIsImF1ZCI6IkFwcERfQVBJcyIsImp0aSI6InRzVTRTazEtazZGOURrXzlTWGVvS3ciLCJzdWIiOiJtYWdhbmciLCJpZFR5cGUiOiJBUElfQ0xJRU5UIiwiaWQiOiI4MTFkYjUzMi1iYWNjLTQ2YmYtOGI5NC01ZjU2OTdkOTMwMTAiLCJhY2N0SWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJ0bnRJZCI6IjVmYzkxOTg4LWExZjgtNDhlMS04YjIwLTlmZjA4NjRkYmFmNiIsImFjY3ROYW1lIjoicHRmd2RpbmRvbmVzaWEtcHJvZCIsInRlbmFudE5hbWUiOiIiLCJmbW1UbnRJZCI6bnVsbCwiYWNjdFBlcm0iOltdLCJyb2xlSWRzIjpbXSwiaWF0IjoxNzY4MjEwNTg2LCJuYmYiOjE3NjgyMTA0NjYsImV4cCI6MTc3MDgwMjU4NiwidG9rZW5UeXBlIjoiQUNDRVNTIn0.3bq0IN3UL5_ZoFwk1mcKYfs776miYviFN6GmTV1XRa8"
BASE_URL = "https://ptfwdindonesia-prod.saas.appdynamics.com"
APP_NAME = "SmartNano"
BT_CSV_FILE = 'business_transactions.csv'

# Target Context
TIER = 'integration-service'
BT = '/smart-integration/users'

# Output Directory
MODEL_DIR = 'model'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_bt_csv():
    """Load business transactions CSV into a lookup dictionary."""
    if not os.path.exists(BT_CSV_FILE):
        return pd.DataFrame()
    try:
        # Use only the first 4 columns
        df = pd.read_csv(BT_CSV_FILE, header=0, usecols=range(4))
        df.columns = ['Tier', 'BT', 'MetricType', 'URL']
        # Clean whitespace
        df['Tier'] = df['Tier'].str.strip()
        df['BT'] = df['BT'].str.strip()
        df['MetricType'] = df['MetricType'].str.strip()
        return df
    except Exception as e:
        print(f"Error loading BT CSV: {e}")
        return pd.DataFrame()

def get_bt_url(tier: str, bt: str, metric_type: str) -> str:
    df = load_bt_csv()
    if df.empty:
        return None
    
    tier = tier.strip()
    bt = bt.strip()
    metric_type = metric_type.strip()
    
    mask = (df['Tier'] == tier) & (df['BT'] == bt) & (df['MetricType'] == metric_type)
    matches = df[mask]
    
    if matches.empty:
        print(f"No URL found for tier='{tier}', bt='{bt}', metric_type='{metric_type}'")
        return None
    
    return matches.iloc[0]['URL']

def fetch_from_appdynamics(url: str, start_time: int, end_time: int) -> list:
    """Fetch data from AppDynamics with explicit start/end times."""
    if not url:
        return []
    
    # Force Time Range
    if 'time-range-type=' in url:
        url = re.sub(r'time-range-type=[^&]+', 'time-range-type=BETWEEN_TIMES', url)
    else:
        url += '&time-range-type=BETWEEN_TIMES'
        
    if 'start-time=' in url:
        url = re.sub(r'start-time=\d+', f'start-time={start_time}', url)
    else:
        url += f'&start-time={start_time}'
        
    if 'end-time=' in url:
        url = re.sub(r'end-time=\d+', f'end-time={end_time}', url)
    else:
        url += f'&end-time={end_time}'
        
    # Remove duration
    url = re.sub(r'&duration-in-mins=\d+', '', url)
    url = re.sub(r'duration-in-mins=\d+&?', '', url)
    
    # Ensure JSON
    if 'output=JSON' not in url:
        url += '&output=JSON'
        
    # Force No Rollup for maximum granularity? 
    # Actually for 1 year, we MIGHT want rollup, otherwise data is massive.
    # But user wants to train models. 1 min granularity for 1 year = 525,600 points.
    # AppDynamics might auto-rollup to 1 hour or something for such wide ranges unless we page.
    # Let's try with rollup=false first, if it fails or is truncated, we might need to loop.
    if 'rollup=false' not in url:
        url += '&rollup=false'
        
    print(f"Fetching: {url}")
    
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {ACCESS_TOKEN}")
    
    try:
        with urllib.request.urlopen(req, timeout=120) as response: # Increased timeout
            raw_data = json.loads(response.read().decode())
             # Flatten logic as per app.py
            final_values = []
            if raw_data and isinstance(raw_data, list):
                for item in raw_data:
                    if 'metricValues' in item:
                        final_values.extend(item['metricValues'])
            return final_values
    except Exception as e:
        print(f"Error fetching: {e}")
        return []

def main():
    # Date Range: 2025 Full Year
    # start_date = datetime(2025, 1, 1, 0, 0, 0)
    # end_date = datetime(2025, 12, 31, 23, 59, 59)
    # Using specific ISO strings directly or just manually creating timestamps
    
    # 2025-01-01 00:00:00 UTC? Or User Local? 
    # AppDynamics usually expects ms timestamp.
    # Let's assume UTC for consistency, or WIB since user is likely Indonesia (PT FWD).
    # app.py uses `.replace('Z', '+00:00')` suggesting UTC or explicit offset.
    # Let's use simple UTC for 2025-01-01 to 2025-12-31 to be safe/standard.
    
    start_dt = datetime(2025, 1, 1, 0, 0, 0)
    end_dt = datetime(2025, 12, 31, 23, 59, 59)
    
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    print(f"Exporting data from {start_dt} to {end_dt}...")
    
    metrics = ['Load', 'Response', 'Error', 'Slow']
    
    for metric in metrics:
        print(f"\n--- Processing {metric} ---")
        url = get_bt_url(TIER, BT, metric)
        if not url:
            print(f"Skipping {metric} (No URL)")
            continue
            
        data = fetch_from_appdynamics(url, start_time=start_ms, end_time=end_ms)
        
        if not data:
            print(f"No data found for {metric}")
            continue
            
        # Process and Save
        filename = os.path.join(MODEL_DIR, f"{metric.lower()}.csv")
        
        # Prepare for CSV
        csv_rows = []
        for val in data:
            # Extract relevant fields
            ts = val.get('startTimeInMillis')
            # Handle different value keys
            value = val.get('value', val.get('sum', 0)) # default to 0
            
            # Format timestamp to readable string for convenience, but keep raw for ML
            dt_str = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
            
            csv_rows.append({
                'timestamp_ms': ts,
                'datetime': dt_str,
                'value': value
            })
            
        # Write to CSV
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            df.sort_values('timestamp_ms', inplace=True)
            df.to_csv(filename, index=False)
            print(f"Saved {len(df)} rows to {filename}")
        else:
            print(f"No rows generated for {metric}")

if __name__ == "__main__":
    main()
