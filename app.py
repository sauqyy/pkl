from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import os
from collections import Counter
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Path to the data file
DATA_FILE = 'metric-data.json'

import urllib.request
import urllib.parse

# Configuration
ACCESS_TOKEN = "eyJraWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBcHBEeW5hbWljcyIsImF1ZCI6IkFwcERfQVBJcyIsImp0aSI6InRzVTRTazEtazZGOURrXzlTWGVvS3ciLCJzdWIiOiJtYWdhbmciLCJpZFR5cGUiOiJBUElfQ0xJRU5UIiwiaWQiOiI4MTFkYjUzMi1iYWNjLTQ2YmYtOGI5NC01ZjU2OTdkOTMwMTAiLCJhY2N0SWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJ0bnRJZCI6IjVmYzkxOTg4LWExZjgtNDhlMS04YjIwLTlmZjA4NjRkYmFmNiIsImFjY3ROYW1lIjoicHRmd2RpbmRvbmVzaWEtcHJvZCIsInRlbmFudE5hbWUiOiIiLCJmbW1UbnRJZCI6bnVsbCwiYWNjdFBlcm0iOltdLCJyb2xlSWRzIjpbXSwiaWF0IjoxNzY4MjEwNTg2LCJuYmYiOjE3NjgyMTA0NjYsImV4cCI6MTc3MDgwMjU4NiwidG9rZW5UeXBlIjoiQUNDRVNTIn0.3bq0IN3UL5_ZoFwk1mcKYfs776miYviFN6GmTV1XRa8"
BASE_URL = "https://ptfwdindonesia-prod.saas.appdynamics.com"
APP_NAME = "SmartNano"
METRIC_PATH = "Overall Application Performance|Average Response Time (ms)"

# Email Configuration (Placeholders - Please Update)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"
RECIPIENT_EMAIL = "your_email@gmail.com"

def send_email_alert(error_message):
    """Sends an email alert when an error occurs."""
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = f"Alert: AppDynamics API Error - {APP_NAME}"
        
        body = f"An error occurred while fetching data from AppDynamics:\n\n{error_message}"
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("Email alert sent successfully.")
    except Exception as e:
        print(f"Failed to send email alert: {e}")

def load_data(duration=60):
    """
    Fetches live data from AppDynamics API based on duration.
    Falls back to local file if API fails.
    """
    encoded_metric = urllib.parse.quote(METRIC_PATH)
    url = (
        f"{BASE_URL}/controller/rest/applications/{APP_NAME}/metric-data?"
        f"metric-path={encoded_metric}&"
        f"time-range-type=BEFORE_NOW&"
        f"duration-in-mins={duration}&"
        f"output=JSON&"
        f"rollup=false"
    )
    
    print(f"Fetching live data for last {duration} minutes...")
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
        error_msg = f"API Error: {e}"
        print(f"{error_msg}. Falling back to cached file.")
        send_email_alert(error_msg)
        
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
    return render_template('executive_dashboard.html')

@app.route('/response-time')
def response_time_dashboard():
    return render_template('dashboard.html')

@app.route('/api/summary')
def get_summary_data():
    summary = {
        'response_time': 0,
        'load': 0,
        'errors': 0,
        'slow_calls': 0
    }
    
    # 1. Response Time (Current/Avg)
    try:
        data = load_data(60) # Last 60 mins
        if data and isinstance(data, list) and len(data) > 0:
             values = data[0].get('metricValues', [])
             if values:
                 summary['response_time'] = round(sum(v['value'] for v in values) / len(values))
    except: pass

    # 2. Errors (Total Last 30 Days)
    try:
        if os.path.exists('error_data.json'):
            with open('error_data.json', 'r') as f:
                errors = json.load(f)
                # Filter last 30 days based on DATA availability to match detailed dashboard
                df = pd.DataFrame(errors)
                if not df.empty:
                    df['dt'] = pd.to_datetime(df['startTimeInMillis'], unit='ms')
                    # Match get_error_analysis logic: Add 7 hours (WIB) and anchor to max date
                    df['dt'] = df['dt'] + timedelta(hours=7) 
                    now_date = df['dt'].max()
                    start_date = now_date - timedelta(days=30)
                    
                    recent_errors = df[df['dt'] >= start_date]
                    summary['errors'] = int(recent_errors['sum'].sum())
    except: pass

    # 3. Load (Calls per Minute)
    # To get Total Calls from CPM: Summing the 'sum' field (which is sum of CPMs per minute) gives Total Calls.
    try:
        if os.path.exists('load_data.json'):
            with open('load_data.json', 'r') as f:
                load_df = pd.DataFrame(json.load(f))
                if not load_df.empty:
                    # Last 30 days sum
                    load_df['dt'] = pd.to_datetime(load_df['startTimeInMillis'], unit='ms')
                    # Consistency: Add 7h and anchor to max
                    load_df['dt'] = load_df['dt'] + timedelta(hours=7)
                    now_load = load_df['dt'].max()
                    start_load = now_load - timedelta(days=30)
                    
                    mask = load_df['dt'] >= start_load
                    # 'sum' field contains the sum of rates (Calls per Minute * Minutes) for Rate metrics
                    summary['load'] = int(load_df[mask]['sum'].sum())
    except: pass
    
    # 4. Slow Calls (Number of Slow Calls)
    # This is a Count metric, so 'sum' or 'value' works, but 'sum' is safer for aggregation.
    try:
        if os.path.exists('slow_calls_data.json'):
            with open('slow_calls_data.json', 'r') as f:
                slow_data = json.load(f)
                slow_df = pd.DataFrame(slow_data)
                if not slow_df.empty:
                    slow_df['dt'] = pd.to_datetime(slow_df['startTimeInMillis'], unit='ms')
                    # Consistency: Add 7h and anchor to max
                    slow_df['dt'] = slow_df['dt'] + timedelta(hours=7)
                    now_slow = slow_df['dt'].max()
                    start_slow = now_slow - timedelta(days=30)
                    
                    mask = slow_df['dt'] >= start_slow
                    summary['slow_calls'] = int(slow_df[mask]['sum'].sum())
    except: pass
    
    return jsonify(summary)

@app.route('/forecasting')
def forecasting():
    return render_template('forecasting.html')

@app.route('/api/data')
def get_data():
    duration = request.args.get('duration', default=60, type=int)
    raw_data = load_data(duration)
    
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

# --- Prediction Logic ---
import torch
import torch.nn as nn
import joblib 
import numpy as np
import pandas as pd

# Define Model Class (Must match training)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

MODEL_PATH = 'lstm_model.pth'
SCALER_PATH = 'scaler.pkl'
TRAINING_DATA = 'training_data.json'

def get_trained_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    
    try:
        # Load Model
        model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        
        # Load Scaler
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

@app.route('/api/forecast')
def get_forecast():
    model, scaler = get_trained_model()
    
    if not model:
        return jsonify({'error': 'Model is not trained yet. Please run train_model.py.'})

    # Prepare Input Data (Last 60 hours)
    # Strategy: Fetch last 3 days of data from AppD -> Resample to Hourly
    # For demo stability, we try to use training_data.json tail if API fails or as simpler source
    
    try:
        # 1. Fetch live data (Last 3 days) to get recent context
        # duration = 4320 mins (3 days)
        # However, for simplicity and speed, let's use the cached training_data.json + whatever live data we can get?
        # Let's just use training_data.json tail for the DEMO to ensure it works immediately.
        
        with open(TRAINING_DATA, 'r') as f:
            raw = json.load(f)
            
        df = pd.DataFrame(raw)
        df = df.sort_values('startTimeInMillis')
        
        # Get tail (last 60)
        last_60 = df.iloc[-60:].copy()
        history_values = last_60['value'].values.astype('float32')
        history_timestamps = last_60['startTimeInMillis'].tolist()
        
        if len(history_values) < 60:
             return jsonify({'error': 'Not enough data for prediction'})
             
        # Scale
        input_data = history_values.reshape(-1, 1)
        scaled_input = scaler.transform(input_data)
        
        # Tensor
        X_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0) # (1, 60, 1)
        
        # Predict Next 24 Steps
        forecasts = []
        current_input = X_tensor
        
        for _ in range(24): # Predict next 24 hours
            with torch.no_grad():
                pred = model(current_input) # (1, 1)
                
            forecasts.append(pred.item())
            
            # Update input: remove first, add new pred
            # pred is (1,1) -> (1, 1, 1)
            new_step = pred.unsqueeze(1)
            current_input = torch.cat((current_input[:, 1:, :], new_step), dim=1)

        # Inverse Scale Predictions
        forecasts = np.array(forecasts).reshape(-1, 1)
        inv_forecasts = scaler.inverse_transform(forecasts).flatten()
        
        # Prepare Response
        last_ts = history_timestamps[-1]
        hour_ms = 3600000
        
        forecast_response = []
        for i, val in enumerate(inv_forecasts):
            last_ts += hour_ms
            forecast_response.append({'timestamp': last_ts, 'value': float(val)})
            
        history_response = []
        for i in range(len(history_values)):
            history_response.append({'timestamp': history_timestamps[i], 'value': float(history_values[i])})
            
        return jsonify({
            'history': history_response,
            'forecast': forecast_response
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})


@app.route('/api/training-status')
def training_status():
    is_training = os.path.exists('training.lock')
    return jsonify({'training': is_training})

@app.route('/erroranalysis')
def error_analysis_page():
    return render_template('erroranalysis.html')

@app.route('/api/error-analysis')
def get_error_analysis():
    error_file = 'error_data.json'
    if not os.path.exists(error_file):
        return jsonify({'error': 'Error data not found. Please run fetch_errors.py first.'})
    
    try:
        with open(error_file, 'r') as f:
            raw_data = json.load(f)
            
        df = pd.DataFrame(raw_data)
        
        # Convert timestamp
        df['dt'] = pd.to_datetime(df['startTimeInMillis'], unit='ms')
        # Adjust for Timezone? Usually UTC. Let's assume UTC for now or add +7 manually if needed.
        # Ideally, we handle TZ in frontend, but for aggregation we need it here.
        # Let's add 7 hours for WIB (Western Indonesia Time) since user is in Indonesia.
        df['dt'] = df['dt'] + pd.Timedelta(hours=7)
        
        # Timeframe Filtering
        timeframe = request.args.get('timeframe', 'all')
        now = df['dt'].max() # Use max date in data as 'now' or actual now? Using data max is safer for historical analysis.
        
        if timeframe == '7d':
             start_date = now - pd.Timedelta(days=7)
             df = df[df['dt'] >= start_date]
        elif timeframe == '30d':
             start_date = now - pd.Timedelta(days=30)
             df = df[df['dt'] >= start_date]
        elif timeframe == '6m':
             start_date = now - pd.Timedelta(days=180)
             df = df[df['dt'] >= start_date]
        elif timeframe == '1y':
             start_date = now - pd.Timedelta(days=365)
             df = df[df['dt'] >= start_date]
             
        df['hour'] = df['dt'].dt.hour
        df['day_name'] = df['dt'].dt.day_name()
        df['day_of_week'] = df['dt'].dt.dayofweek # 0=Monday, 6=Sunday
        
        # filter only non-zero errors?
        # df = df[df['value'] > 0] 
        # Actually we want frequency of errors, so sum of values
        
        # 1. Heatmap Data (Day vs Hour)
        heatmap_data = df.groupby(['day_name', 'hour'])['sum'].sum().reset_index()
        # Pivot for easier frontend consumption
        heatmap_pivot = heatmap_data.pivot(index='day_name', columns='hour', values='sum').fillna(0)
        
        # Reorder days and ensure fillna happens AFTER reindex (if day is missing)
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(days_order).reindex(columns=range(24), fill_value=0).fillna(0)
        
        # 2. Hourly Distribution (Total errors by hour of day)
        # MUST reindex to 0..23 to ensure all hours exist
        hourly_dist = df.groupby('hour')['sum'].sum().reindex(range(24), fill_value=0).tolist()
        
        # 3. Daily Distribution
        daily_dist = df.groupby('day_name')['sum'].sum().reindex(days_order, fill_value=0).tolist()
        
        # 4. Total Errors
        total_errors = int(df['sum'].sum())
        
        # 5. Peak Time
        peak_hour_idx = np.argmax(hourly_dist)
        peak_hour = f"{peak_hour_idx:02d}:00"
        
        # 6. Peak Day
        peak_day_idx = np.argmax(daily_dist)
        peak_day = days_order[peak_day_idx]
        
        return jsonify({
            'heatmap': heatmap_pivot.to_dict(orient='split'), # {index: [days], columns: [0..23], data: [[...]]}
            'hourly': hourly_dist,
            'daily': daily_dist,
            'total': total_errors,
            'peak_hour': peak_hour,
            'peak_day': peak_day
        })
        
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

@app.route('/loadanalysis')
def load_analysis_page():
    return render_template('loadanalysis.html')

@app.route('/api/load-analysis')
def get_load_analysis():
    data_file = 'load_data.json'
    if not os.path.exists(data_file):
        # Fallback to empty if not ready
        return jsonify({'error': 'Load data (load_data.json) not found. Script is likely still running.', 'total': 0})
    
    try:
        with open(data_file, 'r') as f:
            raw_data = json.load(f)
            
        df = pd.DataFrame(raw_data)
        if df.empty:
             return jsonify({'error': 'No data in load_data.json'})

        # Preprocessing
        df['dt'] = pd.to_datetime(df['startTimeInMillis'], unit='ms') + pd.Timedelta(hours=7)
        
        # Timeframe
        timeframe = request.args.get('timeframe', 'all')
        now = df['dt'].max()
        
        if timeframe == '7d': df = df[df['dt'] >= now - pd.Timedelta(days=7)]
        elif timeframe == '30d': df = df[df['dt'] >= now - pd.Timedelta(days=30)]
        elif timeframe == '6m': df = df[df['dt'] >= now - pd.Timedelta(days=180)]
        elif timeframe == '1y': df = df[df['dt'] >= now - pd.Timedelta(days=365)]
        
        df['hour'] = df['dt'].dt.hour
        df['day_name'] = df['dt'].dt.day_name()
        
        # Aggregation
        # Heatmap
        heatmap_pivot = df.groupby(['day_name', 'hour'])['sum'].sum().reset_index().pivot(index='day_name', columns='hour', values='sum').fillna(0)
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(days_order).reindex(columns=range(24), fill_value=0).fillna(0)
        
        # Hourly
        hourly_dist = df.groupby('hour')['sum'].sum().reindex(range(24), fill_value=0).tolist()
        
        # Daily
        daily_dist = df.groupby('day_name')['sum'].sum().reindex(days_order, fill_value=0).tolist()
        
        # Stats
        total_load = int(df['sum'].sum())
        peak_hour = f"{np.argmax(hourly_dist)}:00"
        peak_day = days_order[np.argmax(daily_dist)]
        
        return jsonify({
            'heatmap': heatmap_pivot.to_dict(orient='split'),
            'hourly': hourly_dist,
            'daily': daily_dist,
            'total': total_load,
            'peak_hour': peak_hour,
            'peak_day': peak_day
        })
        
    except Exception as e:
        print(f"Load Analysis Error: {e}")
        return jsonify({'error': str(e)})

@app.route('/slow-calls-analysis')
def slow_calls_analysis_page():
    return render_template('slow_calls_analysis.html')

@app.route('/api/slow-calls-analysis')
def get_slow_calls_analysis():
    # Helper to load specific file
    metric_type = request.args.get('type', 'slow') # 'slow' or 'veryslow'
    
    if metric_type == 'veryslow':
        data_file = 'very_slow_calls_data.json'
    else:
        data_file = 'slow_calls_data.json'

    if not os.path.exists(data_file):
        # Graceful fallback if one file assumes existence but other doesn't
        return jsonify({'error': f'Data file ({data_file}) not found. Script likely running.', 'total': 0})
    
    try:
        with open(data_file, 'r') as f:
            raw_data = json.load(f)
            
        df = pd.DataFrame(raw_data)
        if df.empty:
             return jsonify({'error': f'No data in {data_file}', 'total': 0})

        # Preprocessing
        df['dt'] = pd.to_datetime(df['startTimeInMillis'], unit='ms') + pd.Timedelta(hours=7)
        
        # Timeframe
        timeframe = request.args.get('timeframe', 'all')
        now = df['dt'].max()
        
        if timeframe == '7d': df = df[df['dt'] >= now - pd.Timedelta(days=7)]
        elif timeframe == '30d': df = df[df['dt'] >= now - pd.Timedelta(days=30)]
        elif timeframe == '6m': df = df[df['dt'] >= now - pd.Timedelta(days=180)]
        elif timeframe == '1y': df = df[df['dt'] >= now - pd.Timedelta(days=365)]
        
        df['hour'] = df['dt'].dt.hour
        df['day_name'] = df['dt'].dt.day_name()
        
        # Aggregation
        # Heatmap
        heatmap_pivot = df.groupby(['day_name', 'hour'])['sum'].sum().reset_index().pivot(index='day_name', columns='hour', values='sum').fillna(0)
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(days_order).reindex(columns=range(24), fill_value=0).fillna(0)
        
        # Hourly
        hourly_dist = df.groupby('hour')['sum'].sum().reindex(range(24), fill_value=0).tolist()
        
        # Daily
        daily_dist = df.groupby('day_name')['sum'].sum().reindex(days_order, fill_value=0).tolist()
        
        # Stats (Total is sum of ALL events in period, not count of entries)
        total_calls = int(df['sum'].sum())
        peak_hour = f"{np.argmax(hourly_dist)}:00"
        peak_day = days_order[np.argmax(daily_dist)]
        
        # New Visuals Logic
        # 1. Trend Analysis (Daily Sum)
        # Group by Date (YYYY-MM-DD)
        df['date_str'] = df['dt'].dt.strftime('%Y-%m-%d')
        daily_trend = df.groupby('date_str')['sum'].sum().reset_index()
        trend_data = {
            'labels': daily_trend['date_str'].tolist(),
            'values': daily_trend['sum'].tolist()
        }
        
        # 2. Business Hours Impact
        # Business Hours: 08:00 to 18:00 (inclusive of 08, exclusive of 18? Let's say 8-18)
        # actually 08:00 to 17:59 usually. Let's say hour >= 8 and hour < 18
        business_hours_mask = (df['hour'] >= 8) & (df['hour'] < 18)
        business_calls = df[business_hours_mask]['sum'].sum()
        off_hours_calls = total_calls - business_calls
        
        impact_data = {
            'Business Hours (8-18)': int(business_calls),
            'Off-Hours': int(off_hours_calls)
        }
        
        return jsonify({
            'heatmap': heatmap_pivot.to_dict(orient='split'),
            'hourly': hourly_dist,
            'daily': daily_dist,
            'total': total_calls,
            'peak_hour': peak_hour,
            'peak_day': peak_day,
            'trend': trend_data,
            'impact': impact_data
        })
        
    except Exception as e:
        print(f"Slow Calls Analysis Error: {e}")
        return jsonify({'error': str(e)})

@app.route('/jvm-health')
def jvm_health_page():
    return render_template('jvm_health.html')

@app.route('/api/jvm-data')
def get_jvm_data():
    duration = request.args.get('duration', default=60, type=int) # Default 1 hour for quick view
    
    # Define Metric Paths based on user request
    METRICS = {
        'availability': "Application Infrastructure Performance|dynamic-letter-deployment|Agent|App|Availability",
        'heap_used': "Application Infrastructure Performance|dynamic-letter-deployment|Individual Nodes|dynamic-letter-deployment--18|JVM|Memory|Heap|Used %",
        'heap_used_mb': "Application Infrastructure Performance|dynamic-letter-deployment|Individual Nodes|dynamic-letter-deployment--18|JVM|Memory|Heap|Current Usage (MB)",
        'gc_time': "Application Infrastructure Performance|dynamic-letter-deployment|Individual Nodes|dynamic-letter-deployment--18|JVM|Garbage Collection|Major Collection Time Spent Per Min (ms)",
        'threads_live': "Application Infrastructure Performance|dynamic-letter-deployment|Individual Nodes|dynamic-letter-deployment--18|JVM|Threads|Current No. of Threads",
        'cpu_busy': "Application Infrastructure Performance|dynamic-letter-deployment|Individual Nodes|dynamic-letter-deployment--18|Hardware Resources|CPU|%Busy"
    }
    
    results = {}
    
    # Helper to fetch individual metric (reusing existing auth logic would be best, but let's inline or reuse load_data logic)
    # Since load_data uses a global METRIC_PATH, we need a flexible version.
    
    def fetch_metric_flexible(metric_path, duration_mins):
        encoded_metric = urllib.parse.quote(metric_path)
        # Use 'Smart2' as seen in user provided links for JVM metrics
        # If global APP_NAME is SmartNano, we should probably switch this specific call to Smart2
        target_app = "Smart2" 
        
        url = (
            f"{BASE_URL}/controller/rest/applications/{target_app}/metric-data?"
            f"metric-path={encoded_metric}&"
            f"time-range-type=BEFORE_NOW&"
            f"duration-in-mins={duration_mins}&"
            f"output=JSON&"
            f"rollup=false"
        )
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {ACCESS_TOKEN}")
        try:
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            print(f"Error fetching {metric_path}: {e}")
            return []

    # Fetch all metrics
    # Note: Sequential fetching might be slow. In prod, use ThreadPool. for now, sequential is fine.
    for key, path in METRICS.items():
        data = fetch_metric_flexible(path, duration)
        
        # Process data into simple Time/Value pairs
        series = []
        if data and isinstance(data, list):
             for item in data:
                 if 'metricValues' in item:
                     for val in item['metricValues']:
                         series.append({
                             't': val.get('startTimeInMillis', 0),
                             'v': val.get('value', 0)
                         })
        results[key] = series

    # --- Generate Insights ---
    insights = []
    
    # 1. Availability Insight
    avail_series = results['availability']
    if avail_series:
        avg_avail = sum(d['v'] for d in avail_series) / len(avail_series)
        uptime_percentage = round(avg_avail * 100, 2) if avg_avail <= 1 else round(avg_avail, 2) # Usually 1 or 0, or %? AppD availability usually 1/0
        
        # Check if AppD returns 1 (up) or 0 (down). It seems to be 1=Up.
        # If it's returning %, adjust. Assuming 1=100% based on "Availability".
        
        if avg_avail < 0.99:
            insights.append({'type': 'critical', 'msg': f'Availability is low ({avg_avail:.2%}). System has experienced downtime.'})
        else:
             insights.append({'type': 'success', 'msg': 'System Availability is excellent.'})
    
    # 2. Memory Leak Detection (Simple Trend)
    heap_series = results['heap_used']
    if len(heap_series) > 10:
        # Check first 20% vs last 20% average
        split = len(heap_series) // 5
        start_avg = sum(d['v'] for d in heap_series[:split]) / split
        end_avg = sum(d['v'] for d in heap_series[-split:]) / split
        
        if end_avg > start_avg * 1.2: # 20% growth
            insights.append({'type': 'warning', 'msg': 'Potential Memory Leak detected. Heap usage has increased significantly over time without full recovery.'})
        
        # Critical Threshold
        max_heap = max(d['v'] for d in heap_series)
        if max_heap > 90:
             insights.append({'type': 'critical', 'msg': f'Critical Heap Usage detected ({max_heap}%). Risk of OutOfMemoryError.'})

    # 3. GC Storm Detection
    gc_series = results['gc_time']
    if gc_series:
        high_gc_events = [d for d in gc_series if d['v'] > 5000] # > 5000ms (5s) spent in GC per min is bad
        if len(high_gc_events) > 3:
             insights.append({'type': 'warning', 'msg': f'Detected {len(high_gc_events)} Major GC spikes (>5s). This causes application pauses ("Stop-the-world").'})

    # 4. CPU High Load
    cpu_series = results.get('cpu_busy', [])
    if cpu_series:
        high_cpu = [d for d in cpu_series if d['v'] > 80]
        if len(high_cpu) > 5:
             insights.append({'type': 'warning', 'msg': f'High CPU Usage detected (>80%) for sustained period.'})
             
    # 5. Thread Exhaustion Risk
    thread_series = results.get('threads_live', [])
    if thread_series:
        max_threads = max(d['v'] for d in thread_series)
        if max_threads > 200: # Arbitrary threshold, adjust per app
             insights.append({'type': 'warning', 'msg': f'High Thread Count ({max_threads}). Check for stuck threads or connection pool leaks.'})

    return jsonify({
        'data': results,
        'insights': insights
    })
@app.route('/business-transactions')
def business_transactions_page():
    return render_template('business_transactions.html')

@app.route('/api/business-transactions')
def get_business_transactions():
    file_path = 'List FWD.xlsx'
    sheet_name = 'business_transactions'
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Excel file not found.'})
        
    try:
        # Read with header=None to find the correct header row dynamically
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # Find header row
        header_idx = -1
        for i, row in df_raw.iterrows():
            # Check for key columns. 'Name' and 'Response Time (ms)' seem distinct enough.
            row_values = [str(x).strip() for x in row.values]
            if 'Name' in row_values and 'Response Time (ms)' in row_values:
                header_idx = i
                break
        
        if header_idx == -1:
             return jsonify({'error': 'Could not locate header row with "Name" and "Response Time (ms)" columns.'})

        # Reload or slice
        # Slicing is faster
        df = df_raw.iloc[header_idx + 1:].copy()
        df.columns = [str(x).strip() for x in df_raw.iloc[header_idx]]
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        
        # Clean Data
        required_cols = ['Name', 'Health', 'Response Time (ms)', 'Calls', '% Errors']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
             return jsonify({'error': f'Missing columns: {missing_cols}. Verified columns: {df.columns.tolist()}'})

        # Helper to clean numeric columns
        def clean_numeric(val):
            if pd.isna(val): return 0
            if isinstance(val, (int, float)): return val
            s = str(val).replace(',', '').replace('%', '').replace('-', '0').strip()
            try:
                return float(s)
            except:
                return 0

        df['Response Time (ms)'] = df['Response Time (ms)'].apply(clean_numeric)
        df['Calls'] = df['Calls'].apply(clean_numeric)
        df['% Errors'] = df['% Errors'].apply(clean_numeric)
        
        # Aggregations / Visualizations Data
        
        # 1. Health Distribution
        health_counts = df['Health'].value_counts().to_dict()
        
        # 2. Top 10 Slowest (Response Time)
        top_slowest = df.nlargest(10, 'Response Time (ms)')[['Name', 'Response Time (ms)']]
        
        # 3. Top 10 High Volume (Calls)
        top_volume = df.nlargest(10, 'Calls')[['Name', 'Calls']]
        
        # 4. Top 10 Error Rates (% Errors > 0)
        top_errors = df[df['% Errors'] > 0].nlargest(10, '% Errors')[['Name', '% Errors']]
        
        # 5. Scatter Data: Calls vs Response Time (with Health)
        # Handle NaN in Health just in case
        df['Health'] = df['Health'].fillna('Unknown')
        scatter_data = df[['Name', 'Calls', 'Response Time (ms)', 'Health']].to_dict(orient='records')

        # 6. Table Data
        table_data = df.fillna('').to_dict(orient='records')
        
        return jsonify({
            'health_counts': health_counts,
            'top_slowest': {
                'labels': top_slowest['Name'].tolist(),
                'values': top_slowest['Response Time (ms)'].tolist() 
            },
            'top_volume': {
                'labels': top_volume['Name'].tolist(),
                'values': top_volume['Calls'].tolist()
            },
            'top_errors': {
                'labels': top_errors['Name'].tolist(),
                'values': top_errors['% Errors'].tolist()
            },
            'scatter': scatter_data,
            'table': table_data
        })
        
    except Exception as e:
        print(f"Business Transactions Error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
