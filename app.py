from flask import Flask, render_template, jsonify
import json
import os
from collections import Counter

app = Flask(__name__)

# Path to the data file
DATA_FILE = 'metric-data.json'

import urllib.request
import urllib.parse

# Configuration
ACCESS_TOKEN = "eyJraWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBcHBEeW5hbWljcyIsImF1ZCI6IkFwcERfQVBJcyIsImp0aSI6InRzVTRTazEtazZGOURrXzlTWGVvS3ciLCJzdWIiOiJtYWdhbmciLCJpZFR5cGUiOiJBUElfQ0xJRU5UIiwiaWQiOiI4MTFkYjUzMi1iYWNjLTQ2YmYtOGI5NC01ZjU2OTdkOTMwMTAiLCJhY2N0SWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJ0bnRJZCI6IjVmYzkxOTg4LWExZjgtNDhlMS04YjIwLTlmZjA4NjRkYmFmNiIsImFjY3ROYW1lIjoicHRmd2RpbmRvbmVzaWEtcHJvZCIsInRlbmFudE5hbWUiOiIiLCJmbW1UbnRJZCI6bnVsbCwiYWNjdFBlcm0iOltdLCJyb2xlSWRzIjpbXSwiaWF0IjoxNzY4MjEwNTg2LCJuYmYiOjE3NjgyMTA0NjYsImV4cCI6MTc3MDgwMjU4NiwidG9rZW5UeXBlIjoiQUNDRVNTIn0.3bq0IN3UL5_ZoFwk1mcKYfs776miYviFN6GmTV1XRa8"
BASE_URL = "https://ptfwdindonesia-prod.saas.appdynamics.com"
APP_NAME = "SmartNano"
METRIC_PATH = "Overall Application Performance|Average Response Time (ms)"

def load_data():
    """
    Fetches live data from AppDynamics API (Last 15 minutes).
    Falls back to local file if API fails.
    """
    encoded_metric = urllib.parse.quote(METRIC_PATH)
    url = (
        f"{BASE_URL}/controller/rest/applications/{APP_NAME}/metric-data?"
        f"metric-path={encoded_metric}&"
        f"time-range-type=BEFORE_NOW&"
        f"duration-in-mins=60&"
        f"output=JSON&"
        f"rollup=false"
    )
    
    print("Fetching live data...")
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {ACCESS_TOKEN}")
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            # Save cache
            with open(DATA_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            return data
    except Exception as e:
        print(f"API Error: {e}. Falling back to cached file.")
        
    # Fallback
    if not os.path.exists(DATA_FILE):
        return []
    
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Error reading local file: {e}")
        return []

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    raw_data = load_data()
    
    # Data structures for visualization
    response_times = []
    timeline = [] # For Line Chart: {time: ms, value: ms}
    
    if raw_data and isinstance(raw_data, list):
        for item in raw_data:
            if 'metricValues' in item:
                for val in item['metricValues']:
                    if 'value' in val:
                        v = val['value']
                        t = val.get('startTimeInMillis', 0)
                        
                        response_times.append(v)
                        timeline.append({'time': t, 'value': v})
    
    # 1. Frequency Distribution (Bar Chart)
    frequency = Counter(response_times)
    sorted_freq = dict(sorted(frequency.items()))
    
    # 2. Performance Buckets (Doughnut Chart)
    # Fast: < 50ms, Normal: 50-100ms, Slow: > 100ms
    buckets = {'Fast (<50ms)': 0, 'Normal (50-100ms)': 0, 'Slow (>100ms)': 0}
    for t in response_times:
        if t < 50:
            buckets['Fast (<50ms)'] += 1
        elif t <= 100:
            buckets['Normal (50-100ms)'] += 1
        else:
            buckets['Slow (>100ms)'] += 1

    # 3. Sort timeline by time
    timeline.sort(key=lambda x: x['time'])

    return jsonify({
        'frequency': sorted_freq,
        'timeline': timeline,
        'buckets': buckets,
        'raw_values': response_times
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
