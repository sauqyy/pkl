
import requests
import json
import time

# Wait for server to be potentially ready (or assume it is running, but user's metadata says it is)
# However, I touched app.py, so flask might be reloading.

print("Testing Prophet Forecast API...")

url = "http://127.0.0.1:5000/api/forecast?metric=Response&model=Prophet"

try:
    # Use a timeout just in case
    response = requests.get(url, timeout=60)
    
    if response.status_code == 200:
        data = response.json()
        
        if 'error' in data:
            print(f"FAILED: API returned error: {data['error']}")
        elif 'forecast' in data and len(data['forecast']) > 0:
            print("SUCCESS: Received forecast data.")
            print(f"Model: {data.get('model', 'Unknown')}")
            print(f"Accuracy: {data.get('accuracy', 'N/A')}")
            print(f"Forecast length: {len(data['forecast'])}")
            
            # Print first prediction
            print(f"First prediction: {data['forecast'][0]}")
        else:
            print("FAILED: Unexpected response structure or empty forecast.")
            print(str(data)[:200]) # Print snippet
    else:
        print(f"FAILED: Status Code {response.status_code}")
        print(response.text[:200])

except Exception as e:
    print(f"FAILED: Connection Error: {e}")
