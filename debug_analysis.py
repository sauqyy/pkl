import pandas as pd
import json
import numpy as np

def main():
    try:
        with open('error_data.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if not data:
        print("No data found")
        return

    df = pd.DataFrame(data)
    
    # Preprocessing
    df['dt'] = pd.to_datetime(df['startTimeInMillis'], unit='ms')
    df['dt'] = df['dt'] + pd.Timedelta(hours=7) # WIB
    df['day_name'] = df['dt'].dt.day_name()
    df['hour'] = df['dt'].dt.hour
    
    # Filter Last 30 Days (matching app.py logic roughly)
    now = df['dt'].max()
    start_date = now - pd.Timedelta(days=30)
    df_30d = df[df['dt'] >= start_date].copy()
    
    print(f"Analyzing data from {start_date} to {now}")
    print(f"Total entries: {len(df_30d)}")
    print(f"Total Errors (Sum): {df_30d['sum'].sum()}")
    
    # Hourly Aggregation
    hourly_counts = df_30d.groupby('hour')['sum'].sum()
    print("\n--- Hourly Totals (Top 5) ---")
    print(hourly_counts.sort_values(ascending=False).head(5))
    
    # Breakdown of the Top Hour
    top_hour = hourly_counts.idxmax()
    print(f"\nBreakdown of Peak Hour ({top_hour}:00):")
    peak_details = df_30d[df_30d['hour'] == top_hour][['dt', 'day_name', 'sum']]
    print(peak_details)
    
    # Breakdown of 22:00
    print(f"\nBreakdown of 22:00:")
    h22_details = df_30d[df_30d['hour'] == 22][['dt', 'day_name', 'sum']]
    print(h22_details)

if __name__ == "__main__":
    main()
