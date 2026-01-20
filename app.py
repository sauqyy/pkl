from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import os
import csv
import hashlib
import time
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

# Business Transactions CSV
BT_CSV_FILE = 'business_transactions.csv'

def load_bt_csv():
    """Load business transactions CSV into a lookup dictionary."""
    if not os.path.exists(BT_CSV_FILE):
        return pd.DataFrame()
    try:
        # Use only the first 4 columns to avoid errors with trailing commas
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
    """
    Look up the AppDynamics URL for a specific tier, business transaction, and metric type.
    metric_type should be one of: 'Load', 'Response', 'Error', 'Slow', 'Very Slow'
    """
    df = load_bt_csv()
    if df.empty:
        return None
    
    # Normalize inputs
    tier = tier.strip()
    bt = bt.strip()
    metric_type = metric_type.strip()
    
    mask = (df['Tier'] == tier) & (df['BT'] == bt) & (df['MetricType'] == metric_type)
    matches = df[mask]
    
    if matches.empty:
        print(f"No URL found for tier='{tier}', bt='{bt}', metric_type='{metric_type}'")
        return None
    
    return matches.iloc[0]['URL']

def fetch_from_appdynamics(url: str, duration_override: int = None) -> list:
    """
    Fetch data from an AppDynamics URL.
    Optionally override the duration-in-mins parameter.
    """
    if not url:
        return []
    
    # Override duration if specified
    if duration_override:
        # Parse and update duration in URL
        if 'duration-in-mins=' in url:
            import re
            url = re.sub(r'duration-in-mins=\d+', f'duration-in-mins={duration_override}', url)
    
    # Add output=JSON and rollup=false if not present
    if 'output=JSON' not in url:
        url += '&output=JSON'
    if 'rollup=false' not in url:
        url += '&rollup=false'
    
    print(f"Fetching from AppDynamics: {url[:100]}...")
    
    # Caching Logic
    CACHE_DIR = 'cache'
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{url_hash}.json")
    
    # Check cache (valid for 1 hour)
    if os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < 3600: # 1 hour
            print(f"Loading from cache: {cache_file}")
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading cache: {e}")

    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {ACCESS_TOKEN}")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            
            # Save to cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                print(f"Error writing to cache: {e}")
                
            return data
    except Exception as e:
        print(f"Error fetching from AppDynamics: {e}")
        return []

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
    tier = request.args.get('tier', 'integration-service')
    bt = request.args.get('bt', '/smart-integration/users')
    duration = 43200  # 30 days in minutes
    
    summary = {
        'response_time': 0,
        'load': 0,
        'errors': 0,
        'slow_calls': 0
    }
    
    # 1. Response Time (Average)
    try:
        url = get_bt_url(tier, bt, 'Response')
        if url:
            data = fetch_from_appdynamics(url, 60)  # Last 60 mins for current
            if data and isinstance(data, list) and len(data) > 0:
                values = data[0].get('metricValues', [])
                if values:
                    summary['response_time'] = round(sum(v['value'] for v in values) / len(values))
    except Exception as e:
        print(f"Summary Response Time Error: {e}")

    # 2. Errors (Total Last 30 Days)
    try:
        url = get_bt_url(tier, bt, 'Error')
        if url:
            data = fetch_from_appdynamics(url, duration)
            if data and isinstance(data, list):
                total = 0
                for item in data:
                    if 'metricValues' in item:
                        for v in item['metricValues']:
                            total += v.get('sum', v.get('value', 0))
                summary['errors'] = int(total)
    except Exception as e:
        print(f"Summary Errors Error: {e}")

    # 3. Load (Total Calls Last 30 Days)
    try:
        url = get_bt_url(tier, bt, 'Load')
        if url:
            data = fetch_from_appdynamics(url, duration)
            if data and isinstance(data, list):
                total = 0
                for item in data:
                    if 'metricValues' in item:
                        for v in item['metricValues']:
                            total += v.get('sum', v.get('value', 0))
                summary['load'] = int(total)
    except Exception as e:
        print(f"Summary Load Error: {e}")
    
    # 4. Slow Calls (Total Last 30 Days)
    try:
        url = get_bt_url(tier, bt, 'Slow')
        if url:
            data = fetch_from_appdynamics(url, duration)
            if data and isinstance(data, list):
                total = 0
                for item in data:
                    if 'metricValues' in item:
                        for v in item['metricValues']:
                            total += v.get('sum', v.get('value', 0))
                summary['slow_calls'] = int(total)
    except Exception as e:
        print(f"Summary Slow Calls Error: {e}")
    
    return jsonify(summary)

@app.route('/forecasting')
def forecasting():
    return render_template('forecasting.html')

@app.route('/api/data')
def get_data():
    duration = request.args.get('duration', default=60, type=int)
    tier = request.args.get('tier', 'integration-service')
    bt = request.args.get('bt', '/smart-integration/users')
    
    # Get URL and fetch data
    url = get_bt_url(tier, bt, 'Response')
    if not url:
        # Fallback to old method if no URL found
        raw_data = load_data(duration)
    else:
        raw_data = fetch_from_appdynamics(url, duration)
    
    # Data structures for visualization
    response_times = []
    timeline = []  # For Line Chart: {timestamp: ms, value: ms}
    
    if raw_data and isinstance(raw_data, list):
        for item in raw_data:
            if 'metricValues' in item:
                for val in item['metricValues']:
                    if 'value' in val:
                        v = val['value']
                        t = val.get('startTimeInMillis', 0)
                        
                        response_times.append(v)
                        timeline.append({'timestamp': t, 'value': v})
    
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

    # 3. Sort timeline by timestamp
    timeline.sort(key=lambda x: x['timestamp'])

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
    # Get tier and bt parameters
    tier = request.args.get('tier', 'integration-service')
    bt = request.args.get('bt', '/smart-integration/users')
    timeframe = request.args.get('timeframe', 'all')
    
    # Calculate duration in minutes based on timeframe
    duration_map = {
        '7d': 7 * 24 * 60,
        '30d': 30 * 24 * 60,
        '6m': 180 * 24 * 60,
        '1y': 365 * 24 * 60,
        'all': 43200  # 30 days default for 'all'
    }
    duration = duration_map.get(timeframe, 43200)
    
    # Get URL and fetch data
    url = get_bt_url(tier, bt, 'Error')
    if not url:
        return jsonify({'error': f'No URL found for tier={tier}, bt={bt}, metric_type=Error', 'total': 0})
    
    raw_data = fetch_from_appdynamics(url, duration)
    
    if not raw_data:
        return jsonify({'error': 'Failed to fetch data from AppDynamics', 'total': 0})
    
    try:
        # Extract metric values
        all_values = []
        for item in raw_data:
            if 'metricValues' in item:
                all_values.extend(item['metricValues'])
        
        if not all_values:
            return jsonify({'error': 'No metric values in response', 'total': 0})
            
        df = pd.DataFrame(all_values)
        if df.empty:
            return jsonify({'error': 'Empty dataframe', 'total': 0})
        
        # Preprocessing
        df['dt'] = pd.to_datetime(df['startTimeInMillis'], unit='ms') + pd.Timedelta(hours=7)
        
        df['hour'] = df['dt'].dt.hour
        df['day_name'] = df['dt'].dt.day_name()
        
        # Aggregation - use 'sum' column if available, else 'value'
        value_col = 'sum' if 'sum' in df.columns else 'value'
        
        # Heatmap
        heatmap_pivot = df.groupby(['day_name', 'hour'])[value_col].sum().reset_index().pivot(index='day_name', columns='hour', values=value_col).fillna(0)
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(days_order).reindex(columns=range(24), fill_value=0).fillna(0)
        
        # Hourly
        hourly_dist = df.groupby('hour')[value_col].sum().reindex(range(24), fill_value=0).tolist()
        
        # Daily
        daily_dist = df.groupby('day_name')[value_col].sum().reindex(days_order, fill_value=0).tolist()
        
        # Stats
        total_errors = int(df[value_col].sum())
        peak_hour_idx = np.argmax(hourly_dist)
        peak_hour = f"{peak_hour_idx:02d}:00"
        peak_day_idx = np.argmax(daily_dist)
        peak_day = days_order[peak_day_idx] if daily_dist else 'N/A'
        
        return jsonify({
            'heatmap': heatmap_pivot.to_dict(orient='split'),
            'hourly': hourly_dist,
            'daily': daily_dist,
            'total': total_errors,
            'peak_hour': peak_hour,
            'peak_day': peak_day
        })
        
    except Exception as e:
        print(f"Error Analysis Error: {e}")
        return jsonify({'error': str(e)})

@app.route('/loadanalysis')
def load_analysis_page():
    return render_template('loadanalysis.html')

@app.route('/api/load-analysis')
def get_load_analysis():
    # Get tier and bt parameters
    tier = request.args.get('tier', 'integration-service')
    bt = request.args.get('bt', '/smart-integration/users')
    timeframe = request.args.get('timeframe', 'all')
    
    # Calculate duration in minutes based on timeframe
    duration_map = {
        '7d': 7 * 24 * 60,
        '30d': 30 * 24 * 60,
        '6m': 180 * 24 * 60,
        '1y': 365 * 24 * 60,
        'all': 43200  # 30 days default for 'all'
    }
    duration = duration_map.get(timeframe, 43200)
    
    # Get URL and fetch data
    url = get_bt_url(tier, bt, 'Load')
    if not url:
        return jsonify({'error': f'No URL found for tier={tier}, bt={bt}, metric_type=Load', 'total': 0})
    
    raw_data = fetch_from_appdynamics(url, duration)
    
    if not raw_data:
        return jsonify({'error': 'Failed to fetch data from AppDynamics', 'total': 0})
    
    try:
        # Extract metric values
        all_values = []
        for item in raw_data:
            if 'metricValues' in item:
                all_values.extend(item['metricValues'])
        
        if not all_values:
            return jsonify({'error': 'No metric values in response', 'total': 0})
            
        df = pd.DataFrame(all_values)
        if df.empty:
            return jsonify({'error': 'Empty dataframe', 'total': 0})

        # Preprocessing
        df['dt'] = pd.to_datetime(df['startTimeInMillis'], unit='ms') + pd.Timedelta(hours=7)
        
        df['hour'] = df['dt'].dt.hour
        df['day_name'] = df['dt'].dt.day_name()
        
        # Aggregation - use 'sum' column if available, else 'value'
        value_col = 'sum' if 'sum' in df.columns else 'value'
        
        # Heatmap
        heatmap_pivot = df.groupby(['day_name', 'hour'])[value_col].sum().reset_index().pivot(index='day_name', columns='hour', values=value_col).fillna(0)
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(days_order).reindex(columns=range(24), fill_value=0).fillna(0)
        
        # Hourly
        hourly_dist = df.groupby('hour')[value_col].sum().reindex(range(24), fill_value=0).tolist()
        
        # Daily
        daily_dist = df.groupby('day_name')[value_col].sum().reindex(days_order, fill_value=0).tolist()
        
        # Stats
        total_load = int(df[value_col].sum())
        peak_hour = f"{np.argmax(hourly_dist)}:00"
        peak_day = days_order[np.argmax(daily_dist)] if daily_dist else 'N/A'
        
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
    # Get tier and bt parameters
    tier = request.args.get('tier', 'integration-service')
    bt = request.args.get('bt', '/smart-integration/users')
    timeframe = request.args.get('timeframe', 'all')
    metric_type = request.args.get('type', 'slow')  # 'slow' or 'veryslow'
    
    # Calculate duration in minutes based on timeframe
    duration_map = {
        '7d': 7 * 24 * 60,
        '30d': 30 * 24 * 60,
        '6m': 180 * 24 * 60,
        '1y': 365 * 24 * 60,
        'all': 43200  # 30 days default for 'all'
    }
    duration = duration_map.get(timeframe, 43200)
    
    # Map metric type to CSV metric name
    metric_name = 'Very Slow' if metric_type == 'veryslow' else 'Slow'
    
    # Get URL and fetch data
    url = get_bt_url(tier, bt, metric_name)
    if not url:
        return jsonify({'error': f'No URL found for tier={tier}, bt={bt}, metric_type={metric_name}', 'total': 0})
    
    raw_data = fetch_from_appdynamics(url, duration)
    
    if not raw_data:
        return jsonify({'error': 'Failed to fetch data from AppDynamics', 'total': 0})
    
    try:
        # Extract metric values
        all_values = []
        for item in raw_data:
            if 'metricValues' in item:
                all_values.extend(item['metricValues'])
        
        if not all_values:
            return jsonify({'error': 'No metric values in response', 'total': 0})
            
        df = pd.DataFrame(all_values)
        if df.empty:
            return jsonify({'error': 'Empty dataframe', 'total': 0})

        # Preprocessing
        df['dt'] = pd.to_datetime(df['startTimeInMillis'], unit='ms') + pd.Timedelta(hours=7)
        
        df['hour'] = df['dt'].dt.hour
        df['day_name'] = df['dt'].dt.day_name()
        
        # Aggregation - use 'sum' column if available, else 'value'
        value_col = 'sum' if 'sum' in df.columns else 'value'
        
        # Heatmap
        heatmap_pivot = df.groupby(['day_name', 'hour'])[value_col].sum().reset_index().pivot(index='day_name', columns='hour', values=value_col).fillna(0)
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(days_order).reindex(columns=range(24), fill_value=0).fillna(0)
        
        # Hourly
        hourly_dist = df.groupby('hour')[value_col].sum().reindex(range(24), fill_value=0).tolist()
        
        # Daily
        daily_dist = df.groupby('day_name')[value_col].sum().reindex(days_order, fill_value=0).tolist()
        
        # Stats
        total_calls = int(df[value_col].sum())
        peak_hour = f"{np.argmax(hourly_dist)}:00"
        peak_day = days_order[np.argmax(daily_dist)] if daily_dist else 'N/A'
        
        # Trend Analysis (Daily Sum)
        df['date_str'] = df['dt'].dt.strftime('%Y-%m-%d')
        daily_trend = df.groupby('date_str')[value_col].sum().reset_index()
        trend_data = {
            'labels': daily_trend['date_str'].tolist(),
            'values': daily_trend[value_col].tolist()
        }
        
        # Business Hours Impact
        business_hours_mask = (df['hour'] >= 8) & (df['hour'] < 18)
        business_calls = int(df[business_hours_mask][value_col].sum())
        off_hours_calls = total_calls - business_calls
        
        impact_data = {
            'Business Hours (8-18)': business_calls,
            'Off-Hours': off_hours_calls
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
    duration = request.args.get('duration', default=60, type=int)
    tier = request.args.get('tier', 'integration-service')
    
    # Map tier names to their node identifiers (from the user's tier URLs)
    TIER_NODE_MAP = {
        'integration-service': 'integration-service--1',
        'remote-selling-service-deployment': 'remote-selling-service-deployment--14',
        'dynamic-letter-deployment': 'dynamic-letter-deployment--18',
        'payment-service': 'payment-service--1',
        'validation-ph-service-deployment': 'validation-ph-service-deployment--17',
        'otp-smart2-deployment': 'otp-smart2-deployment--22',
    }
    
    node = TIER_NODE_MAP.get(tier, tier + '--1')
    
    # Define Metric Paths dynamically based on tier
    METRICS = {
        'availability': f"Application Infrastructure Performance|{tier}|Agent|App|Availability",
        'heap_used': f"Application Infrastructure Performance|{tier}|Individual Nodes|{node}|JVM|Memory|Heap|Used %",
        'heap_used_mb': f"Application Infrastructure Performance|{tier}|Individual Nodes|{node}|JVM|Memory|Heap|Current Usage (MB)",
        'gc_time': f"Application Infrastructure Performance|{tier}|Individual Nodes|{node}|JVM|Garbage Collection|Major Collection Time Spent Per Min (ms)",
        'gc_count': f"Application Infrastructure Performance|{tier}|Individual Nodes|{node}|JVM|Garbage Collection|Number of Major Collections Per Min",
        'threads_live': f"Application Infrastructure Performance|{tier}|Individual Nodes|{node}|JVM|Threads|Current No. of Threads",
        'cpu_busy': f"Application Infrastructure Performance|{tier}|Individual Nodes|{node}|Hardware Resources|CPU|%Busy"
    }
    
    results = {}
    
    def fetch_metric_flexible(metric_path, duration_mins):
        encoded_metric = urllib.parse.quote(metric_path)
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
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            print(f"Error fetching {metric_path}: {e}")
            return []

    # Fetch all metrics
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
    avail_series = results.get('availability', [])
    if avail_series:
        avg_avail = sum(d['v'] for d in avail_series) / len(avail_series)
        
        if avg_avail < 0.99:
            insights.append({'type': 'critical', 'msg': f'Availability is low ({avg_avail:.2%}). System has experienced downtime.'})
        else:
            insights.append({'type': 'success', 'msg': f'System Availability for {tier} is excellent.'})
    
    # 2. Memory Leak Detection (Simple Trend)
    heap_series = results.get('heap_used', [])
    if len(heap_series) > 10:
        split = len(heap_series) // 5
        start_avg = sum(d['v'] for d in heap_series[:split]) / split
        end_avg = sum(d['v'] for d in heap_series[-split:]) / split
        
        if end_avg > start_avg * 1.2:
            insights.append({'type': 'warning', 'msg': 'Potential Memory Leak detected. Heap usage has increased significantly over time without full recovery.'})
        
        max_heap = max(d['v'] for d in heap_series)
        if max_heap > 90:
            insights.append({'type': 'critical', 'msg': f'Critical Heap Usage detected ({max_heap}%). Risk of OutOfMemoryError.'})

    # 3. GC Storm Detection
    gc_series = results.get('gc_time', [])
    if gc_series:
        high_gc_events = [d for d in gc_series if d['v'] > 5000]
        if len(high_gc_events) > 3:
            insights.append({'type': 'warning', 'msg': f'Detected {len(high_gc_events)} Major GC spikes (>5s). This causes application pauses ("Stop-the-world").'})

    # 4. CPU High Load
    cpu_series = results.get('cpu_busy', [])
    if cpu_series:
        high_cpu = [d for d in cpu_series if d['v'] > 80]
        if len(high_cpu) > 5:
            insights.append({'type': 'warning', 'msg': 'High CPU Usage detected (>80%) for sustained period.'})
             
    # 5. Thread Exhaustion Risk
    thread_series = results.get('threads_live', [])
    if thread_series:
        max_threads = max(d['v'] for d in thread_series)
        if max_threads > 200:
            insights.append({'type': 'warning', 'msg': f'High Thread Count ({max_threads}). Check for stuck threads or connection pool leaks.'})

    return jsonify({
        'data': results,
        'insights': insights,
        'tier': tier
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
        table_data = df[['Name', 'Health', 'Response Time (ms)', 'Calls', '% Errors']].head(50).to_dict(orient='records')

        return jsonify({
            'health_dist': health_counts,
            'top_slowest': top_slowest.to_dict(orient='records'),
            'top_volume': top_volume.to_dict(orient='records'),
            'top_errors': top_errors.to_dict(orient='records'),
            'scatter': scatter_data,
            'table': table_data
        })

    except Exception as e:
        print(f"Error processing stats: {e}")
        return jsonify({'error': str(e)})

@app.route('/database-analysis')
def database_analysis_page():
    return render_template('database_analysis.html')

@app.route('/api/database-analysis')
def get_database_analysis():
    # 1. Fetch Data
    duration = request.args.get('duration', default=60, type=int)
    
    # Metric Paths
    metric_time_spent = "Databases|C2L_DMS|KPI|Time Spent in Executions (s)"
    metric_load = "Databases|C2L_DMS|KPI|Calls per Minute"
    metric_cpu = "Databases|C2L_DMS|Hardware Resources|CPU|%Busy"
    
    # Helper to fetch data
    def fetch_metric(m_path):
        encoded_metric = urllib.parse.quote(m_path)
        target_app = "Database Monitoring"
        url = (
            f"{BASE_URL}/controller/rest/applications/{urllib.parse.quote(target_app)}/metric-data?"
            f"metric-path={encoded_metric}&"
            f"time-range-type=BEFORE_NOW&"
            f"duration-in-mins={duration}&"
            f"output=JSON&"
            f"rollup=false"
        )
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {ACCESS_TOKEN}")
        try:
            with urllib.request.urlopen(req) as response:
                raw = json.loads(response.read().decode())
                if raw and len(raw) > 0 and 'metricValues' in raw[0]:
                    return raw[0]['metricValues']
        except Exception as e:
            print(f"Fetch Error ({m_path}): {e}")
        return []

    # helper to parse queries.csv
    def get_top_queries():
        queries_path = os.path.join(os.path.dirname(__file__), 'queries.csv')
        top_queries = []
        try:
            if os.path.exists(queries_path):
                with open(queries_path, 'r', encoding='utf-8', errors='replace') as f:
                    # Skip first 6 lines of metadata
                    for _ in range(6):
                        next(f)
                    
                    reader = csv.DictReader(f)
                    all_queries = []
                    
                    for row in reader:
                        # Clean up keys (sometimes they have whitespace)
                        clean_row = {k.strip(): v.strip() for k, v in row.items() if k}
                        
                        # Extract Time and Weight
                        weight_str = clean_row.get('Weight (%)', '0').replace('%', '')
                        weight = float(weight_str) if weight_str else 0.0
                        
                        all_queries.append({
                            'id': clean_row.get('Query Id', 'N/A'),
                            'query': clean_row.get('Query', 'N/A'),
                            'elapsed_time': clean_row.get('Elapsed Time', 'N/A'),
                            'executions': clean_row.get('Number of Executions', '0'),
                            'avg_response': clean_row.get('Average Response Time', 'N/A'),
                            'weight': weight
                        })
                    
                    # Sort by Weight descending
                    all_queries.sort(key=lambda x: x['weight'], reverse=True)
                    top_queries = all_queries[:10]
        except Exception as e:
            print(f"Error parsing queries.csv: {e}")
            
        return top_queries

    data_time = fetch_metric(metric_time_spent)
    data_load = fetch_metric(metric_load)
    data_cpu = fetch_metric(metric_cpu)
    queries_data = get_top_queries() # Function name kept for simplicity, but returns all now

    # 2. Process & Align Data
    # Use a dictionary to align by timestamp: {timestamp: {'time': v1, 'load': v2, 'cpu': v3}}
    aligned_data = {}
    
    for item in data_time:
        ts = item.get('startTimeInMillis', 0)
        val = item.get('value', 0)
        if ts not in aligned_data: aligned_data[ts] = {'time': 0, 'load': 0, 'cpu': 0}
        aligned_data[ts]['time'] = val
        
    for item in data_load:
        ts = item.get('startTimeInMillis', 0)
        val = item.get('value', 0)
        if ts not in aligned_data: aligned_data[ts] = {'time': 0, 'load': 0, 'cpu': 0}
        aligned_data[ts]['load'] = val

    for item in data_cpu:
        ts = item.get('startTimeInMillis', 0)
        val = item.get('value', 0)
        if ts not in aligned_data: aligned_data[ts] = {'time': 0, 'load': 0, 'cpu': 0}
        aligned_data[ts]['cpu'] = val
        
    # Sort by timestamp
    sorted_ts = sorted(aligned_data.keys())
    
    timestamps = []
    values_time = []
    values_load = []
    values_cpu = []
    
    inefficiency_count = 0
    total_points = len(sorted_ts)
    
    for ts in sorted_ts:
        t_val = aligned_data[ts]['time']
        l_val = aligned_data[ts]['load']
        c_val = aligned_data[ts]['cpu']
        
        timestamps.append(ts)
        values_time.append(t_val)
        values_load.append(l_val)
        values_cpu.append(c_val)
        
        # Analysis: Time Spent > Load
        if t_val > l_val:
            inefficiency_count += 1

    if total_points == 0:
        return jsonify({'error': 'No data received from AppDynamics'})

    return jsonify({
        'chart_data': { 
            'timestamps': timestamps, 
            'time_spent': values_time,
            'load': values_load,
            'cpu': values_cpu
        },
        'inefficiency_count': inefficiency_count,
        'total_points': total_points,
        'queries': queries_data
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
