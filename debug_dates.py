import json
import pandas as pd
from datetime import datetime

files = ['error_data.json', 'load_data.json', 'slow_calls_data.json', 'very_slow_calls_data.json']

print(f"Current System Time: {datetime.now()}")

for fpath in files:
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            if not df.empty and 'startTimeInMillis' in df.columns:
                min_ts = df['startTimeInMillis'].min()
                max_ts = df['startTimeInMillis'].max()
                print(f"\nFile: {fpath}")
                print(f"  Min Date: {pd.to_datetime(min_ts, unit='ms')}")
                print(f"  Max Date: {pd.to_datetime(max_ts, unit='ms')}")
                print(f"  Count: {len(df)}")
            else:
                print(f"\nFile: {fpath} - Empty or wrong format")
    except Exception as e:
        print(f"\nFile: {fpath} - Error: {e}")
