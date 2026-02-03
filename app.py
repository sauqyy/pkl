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
from datetime import datetime, timedelta
import requests
import threading
import re
import logging
from logging.handlers import RotatingFileHandler

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "8290043825:AAFeZVa2F8kBXJduCktSJOJ162SRF9QDhFM"
TELEGRAM_CHAT_ID = "5958836175" # Update this with your Chat ID

# --- Monitor Logger Setup ---
monitor_logger = logging.getLogger('monitor_logger')
monitor_logger.setLevel(logging.INFO)
if not monitor_logger.handlers:
    # File Handler
    file_handler = RotatingFileHandler('monitor.log', maxBytes=2*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    monitor_logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    monitor_logger.addHandler(console_handler)

def send_telegram_alert(message):
    """Sends a Telegram alert when an error occurs."""
    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        print("Telegram Chat ID not configured. Skipping alert.")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": f"🚨 *AppDynamics Alert*\n\n{message}",
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=data)
        if response.status_code != 200:
            print(f"Failed to send Telegram alert: {response.text}")
    except Exception as e:
        print(f"Error sending Telegram alert: {e}")


app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Path to the data file
DATA_FILE = 'metric-data.json'

import urllib.request
import urllib.parse

# Configuration
ACCESS_TOKEN = "eyJraWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBcHBEeW5hbWljcyIsImF1ZCI6IkFwcERfQVBJcyIsImp0aSI6InRzVTRTazEtazZGOURrXzlTWGVvS3ciLCJzdWIiOiJtYWdhbmciLCJpZFR5cGUiOiJBUElfQ0xJRU5UIiwiaWQiOiI4MTFkYjUzMi1iYWNjLTQ2YmYtOGI5NC01ZjU2OTdkOTMwMTAiLCJhY2N0SWQiOiI1ZmM5MTk4OC1hMWY4LTQ4ZTEtOGIyMC05ZmYwODY0ZGJhZjYiLCJ0bnRJZCI6IjVmYzkxOTg4LWExZjgtNDhlMS04YjIwLTlmZjA4NjRkYmFmNiIsImFjY3ROYW1lIjoicHRmd2RpbmRvbmVzaWEtcHJvZCIsInRlbmFudE5hbWUiOiIiLCJmbW1UbnRJZCI6bnVsbCwiYWNjdFBlcm0iOltdLCJyb2xlSWRzIjpbXSwiaWF0IjoxNzY4MjEwNTg2LCJuYmYiOjE3NjgyMTA0NjYsImV4cCI6MTc3MDgwMjU4NiwidG9rZW5UeXBlIjoiQUNDRVNTIn0.3bq0IN3UL5_ZoFwk1mcKYfs776miYviFN6GmTV1XRa8"
BASE_URL = "https://ptfwdindonesia-prod.saas.appdynamics.com"
APP_NAME = "Smart2"
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

def get_time_params(default_duration=60):
    """
    Utility to get duration and start/end timestamps from request args.
    Used consistently across all API endpoints.
    """
    duration = request.args.get('duration', default=default_duration, type=int)
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    start_time_ms = None
    end_time_ms = None

    if start_date_str and end_date_str:
        try:
            if start_date_str.endswith('Z'):
                start_date_str = start_date_str.replace('Z', '+00:00')
            if end_date_str.endswith('Z'):
                end_date_str = end_date_str.replace('Z', '+00:00')
                
            s_dt = datetime.fromisoformat(start_date_str)
            e_dt = datetime.fromisoformat(end_date_str)
            
            start_time_ms = int(s_dt.timestamp() * 1000)
            end_time_ms = int(e_dt.timestamp() * 1000)
            
            diff = e_dt - s_dt
            calculated_mins = int(diff.total_seconds() / 60)
            if calculated_mins > 0:
                duration = calculated_mins
        except Exception as e:
            print(f"Error parsing dates: {e}")
            
    return duration, start_time_ms, end_time_ms

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

def fetch_from_appdynamics(url: str, duration: int = None, start_time: int = None, end_time: int = None, allow_rollup: bool = False) -> list:
    """
    Fetch data from an AppDynamics URL.
    Optionally override the duration-in-mins parameter OR use absolute start/end times (ms).
    Allow rollup for large datasets to improve performance.
    """
    if not url:
        return []
    
    # Override time range if specified
    if start_time and end_time:
        # Use absolute time range
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
            
        # Remove duration if it exists to avoid confusion
        url = re.sub(r'&duration-in-mins=\d+', '', url)
        url = re.sub(r'duration-in-mins=\d+&?', '', url)

    elif duration:
        # Parse and update duration in URL (relative time)
        if 'duration-in-mins=' in url:
            url = re.sub(r'duration-in-mins=\d+', f'duration-in-mins={duration}', url)
        else:
            url += f'&duration-in-mins={duration}'
        
        # Ensure it's BEFORE_NOW if duration is given without start/end
        if 'time-range-type=' in url:
            url = re.sub(r'time-range-type=[^&]+', 'time-range-type=BEFORE_NOW', url)
    
    print(f"DEBUG FETCH URL (Start={start_time}, End={end_time}, Duration={duration}): {url}")
    
    # Add output=JSON
    if 'output=JSON' not in url:
        url += '&output=JSON'
        
    # Handle Rollup
    if not allow_rollup:
        if 'rollup=false' not in url:
            url += '&rollup=false'
    
    # Caching Logic
    CACHE_DIR = 'cache'
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{url_hash}.json")
    
    raw_data = None
    
    # Check cache
    # Check cache
    if os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        
        # Smart TTL based on duration
        ttl = 3600 # default 1 hour
        if duration and duration <= 15:
            ttl = 60 # 1 minute for short windows
            
        if file_age < ttl:
            # print(f"Loading from cache: {cache_file}")
            try:
                with open(cache_file, 'r') as f:
                    raw_data = json.load(f)
            except Exception as e:
                print(f"Error reading cache: {e}")

    # Fetch from Network if no cache
    if raw_data is None:
        # print(f"Fetching from AppDynamics: {url[:150]}...")
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {ACCESS_TOKEN}")
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                raw_data = json.loads(response.read().decode())
                
                # Save Raw Data to Cache
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(raw_data, f)
                except Exception as e:
                    print(f"Error writing to cache: {e}")
                    
        except Exception as e:
            error_msg = f"Error fetching from AppDynamics: {e}\nURL: {url}"
            print(error_msg)
            # send_telegram_alert(error_msg) 
            return []


    # Flatten Data (Source of Truth for callers)
    # Callers expect a list of 'metricValues' objects, not the raw API response.
    final_values = []
    if raw_data and isinstance(raw_data, list):
        for item in raw_data:
            if 'metricValues' in item:
                final_values.extend(item['metricValues'])
                
    return final_values

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
        send_telegram_alert(error_msg)
        
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
    
    # Get time params using 7 days default for summary
    duration, start_time_ms, end_time_ms = get_time_params(default_duration=10080)
    
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
            data = fetch_from_appdynamics(url, duration, start_time=start_time_ms, end_time=end_time_ms)
            if data and isinstance(data, list) and len(data) > 0:
                # data is already flattened list of metric values
                summary['response_time'] = round(sum(v.get('value', 0) for v in data) / len(data))
    except Exception as e:
        print(f"Summary Response Time Error: {e}")

    # 2. Errors (Total)
    try:
        url = get_bt_url(tier, bt, 'Error')
        if url:
            data = fetch_from_appdynamics(url, duration, start_time=start_time_ms, end_time=end_time_ms)
            if data and isinstance(data, list):
                total = sum(item.get('sum', item.get('value', 0)) for item in data)
                summary['errors'] = int(total)
    except Exception as e:
        print(f"Summary Errors Error: {e}")

    # 3. Load (Total Calls)
    try:
        url = get_bt_url(tier, bt, 'Load')
        if url:
            data = fetch_from_appdynamics(url, duration, start_time=start_time_ms, end_time=end_time_ms)
            if data and isinstance(data, list):
                total = sum(item.get('sum', item.get('value', 0)) for item in data)
                summary['load'] = int(total)
    except Exception as e:
        print(f"Summary Load Error: {e}")
    
    # 4. Slow Calls (Total)
    try:
        url = get_bt_url(tier, bt, 'Slow')
        if url:
            data = fetch_from_appdynamics(url, duration, start_time=start_time_ms, end_time=end_time_ms)
            if data and isinstance(data, list):
                total = sum(item.get('sum', item.get('value', 0)) for item in data)
                summary['slow_calls'] = int(total)
    except Exception as e:
        print(f"Summary Slow Calls Error: {e}")
    
    return jsonify(summary)

@app.route('/forecasting')
def forecasting():
    return render_template('forecasting.html')

@app.route('/api/data')
def get_data():
    tier = request.args.get('tier', 'integration-service')
    bt = request.args.get('bt', '/smart-integration/users')
    
    # Get time params using 60 mins default
    duration, start_time_ms, end_time_ms = get_time_params(default_duration=60)

    # Get URL and fetch data
    url = get_bt_url(tier, bt, 'Response')
    if not url:
        # Fallback to old method if no URL found
        raw_data = load_data(duration)
    else:
        raw_data = fetch_from_appdynamics(url, duration, start_time=start_time_ms, end_time=end_time_ms)
    
    # Data structures for visualization
    response_times = []
    timeline = []  # For Line Chart: {timestamp: ms, value: ms}
    
    if raw_data and isinstance(raw_data, list):
        for val in raw_data:
             if 'value' in val:
                 ms = val['value']
                 ts = val['startTimeInMillis']
                 
                 # Add to list
                 response_times.append(ms)
                 
                 # Add to timeline
                 # Convert timestamp to HH:MM format
                 import datetime
                 dt_object = datetime.datetime.fromtimestamp(ts / 1000) + datetime.timedelta(hours=7) # WIB
                 time_str = dt_object.strftime("%H:%M")
                 
                 timeline.append({
                     'timestamp': ts,  # Return raw timestamp in ms
                     'time_str': time_str,
                     'value': ms,
                     'full_date': dt_object.isoformat()
                 })
    
    # 1. Frequency Distribution (Bar Chart)
    frequency = Counter(response_times)
    sorted_freq = dict(sorted(frequency.items()))
    
    # 2. Performance Buckets (Doughnut Chart)
    # Fast: < 500ms, Normal: 500-1000ms, Slow: > 1000ms
    buckets = {'Fast (<500ms)': 0, 'Normal (500-1000ms)': 0, 'Slow (>1000ms)': 0}
    for t in response_times:
        if t < 500:
            buckets['Fast (<500ms)'] += 1
        elif t <= 1000:
            buckets['Normal (500-1000ms)'] += 1
        else:
            buckets['Slow (>1000ms)'] += 1

    # 3. Sort timeline by timestamp
    timeline.sort(key=lambda x: x['timestamp'])

    min_response = min([x for x in response_times if x > 0]) if any(x > 0 for x in response_times) else 0
    max_response = max(response_times) if response_times else 0

    return jsonify({
        'frequency': sorted_freq,
        'timeline': timeline,
        'buckets': buckets,
        'raw_values': response_times,
        'min': min_response,
        'max': max_response
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

from train_model import train_metric_model, get_model_paths, ANNAutoencoder, GRUAutoencoder
import prophet_model

def get_trained_model(metric_name='response', model_type='LSTM'):
    """Load the trained model and scaler for a specific metric and type."""
    model_path, scaler_path = get_model_paths(metric_name, model_type)
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    try:
        # Load Model based on type
        if model_type == 'ANN':
            model = ANNAutoencoder(input_dim=60)
        elif model_type == 'GRU':
            model = GRUAutoencoder(input_size=1, hidden_size=16, seq_len=60)
        else: # LSTM
            model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
            
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Load Scaler
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"Error loading {model_type} model for {metric_name}: {e}")
        return None, None

@app.route('/api/forecast')
def get_forecast():
    metric = request.args.get('metric', 'Response')  # Load, Response, Error, Slow
    model_type = request.args.get('model', 'LSTM') # LSTM, ANN, GRU
    tier = request.args.get('tier', 'integration-service')
    bt = request.args.get('bt', '/smart-integration/users')
    
    # ... (date validation remains same)
    valid_metrics = ['Load', 'Response', 'Error', 'Slow']
    if metric not in valid_metrics:
        return jsonify({'error': f'Invalid metric type. Must be one of: {valid_metrics}'})
    
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    start_time = None
    end_time = None
    
    if start_date and end_date:
        try:
            dt_start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            dt_end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            start_time = int(dt_start.timestamp() * 1000)
            end_time = int(dt_end.timestamp() * 1000)
        except ValueError:
            pass

    try:
        # Fetch historical data from AppDynamics (usually 30 days for training/context)
        # But if the user selects a custom range, we use that.
        # Fallback duration for forecast: 30 days
        duration, start_time_ms, end_time_ms = get_time_params(default_duration=43200)

        url = get_bt_url(tier, bt, metric)
        
        if not url:
            return jsonify({'error': f'No URL configured for metric: {metric}'})
        
        # Use our updated central function
        raw_data = fetch_from_appdynamics(url, duration, start_time=start_time_ms, end_time=end_time_ms)
        
        if not raw_data:
            return jsonify({'error': f'Failed to fetch {metric} data from AppDynamics'})
        
        # Extract metric values - raw_data is already a flattened list of metric values
        # from fetch_from_appdynamics, so we just need to extract timestamp and value
        all_values = []
        for val in raw_data:
            all_values.append({
                'timestamp': val.get('startTimeInMillis', 0),
                'value': val.get('value', val.get('sum', 0))
            })
        
        if len(all_values) < 24:
            return jsonify({'error': f'Not enough historical data for {metric} forecast'})
        
        # Sort by timestamp
        all_values.sort(key=lambda x: x['timestamp'])
        
        # Resample to hourly data
        df = pd.DataFrame(all_values)
        df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('dt', inplace=True)
        
        # Resample to hourly, using sum for count-based metrics, mean for response time
        if metric == 'Response':
            hourly = df['value'].resample('h').mean().dropna()
        else:
            hourly = df['value'].resample('h').sum()
        
        hourly = hourly.tail(720)  # Last 30 days of hourly data (720 hours)

        
        if len(hourly) < 24:
            return jsonify({'error': f'Not enough hourly data points for {metric}'})
        
        history_values = hourly.values.astype('float32')
        history_timestamps = [int(ts.timestamp() * 1000) for ts in hourly.index]
        
        # Use LSTM for ALL metrics
        # Try to get specific model for this metric
        model = None
        if model_type != 'Prophet':
            model, _ = get_trained_model(metric, model_type)
        
        lock_file = f'training_{metric}.lock'
        
        # Check if currently training
        if model_type != 'Prophet' and os.path.exists(lock_file):
             # Check lock file age
             lock_age = time.time() - os.path.getmtime(lock_file)
             if lock_age > 600: # 10 minutes timeout
                 try:
                     os.remove(lock_file)
                     print(f"Removed stale lock for {metric}")
                 except:
                     pass
             else:
                 return jsonify({'status': 'training', 'message': f'{model_type} model for {metric} is currently training...'})
             
        # If no model exists and not training, trigger background training
        if model_type != 'Prophet' and not model and len(history_values) >= 100:
             import threading
             
             # Prepare data list for training
             training_data = [{'startTimeInMillis': ts, 'value': val} 
                              for ts, val in zip(history_timestamps, history_values)]
                              
             def run_training_background(data, m_name, m_type):
                 print(f"Background training started for {m_name} ({m_type})...")
                 train_metric_model(data, m_name, m_type)
                 print(f"Background training finished for {m_name}")
                 
             thread = threading.Thread(target=run_training_background, args=(training_data, metric, model_type))
             thread.start()
             
             return jsonify({'status': 'training', 'message': f'Started training {model_type} model for {metric}'})
        
        if model_type == 'Prophet':
            try:
                # Prophet handles its own training/fitting on the fly usually, or we can save/load models.
                # Given strict time constraint or "live" feel, we might retrain on history every time or use a cached model.
                # For this implementation, we will train on the fly as Prophet is relatively fast for small data (720 points).
                
                result = prophet_model.train_predict(history_values, history_timestamps, forecast_days=7, metric_name=metric)
                return jsonify(result)
            except Exception as e:
                print(f"Prophet error: {e}")
                return jsonify({'error': f'Prophet model failed: {str(e)}'})

        if model and len(history_values) >= 84:  # Need 60 for input + 24 for comparison
            try:
                from sklearn.preprocessing import MinMaxScaler
                
                # Create a new scaler fitted to THIS metric's data
                metric_scaler = MinMaxScaler(feature_range=(0, 1))
                input_data = history_values.reshape(-1, 1)
                metric_scaler.fit(input_data)  # Fit on all history
                
                if model_type == 'LSTM':
                    # --- BACKTESTING: Compare what we predicted vs what actually happened ---
                    # Use data from 24 hours ago to predict, then compare with actual
                    backtest_start = len(history_values) - 84  # 60 for input + 24 for comparison
                    backtest_input = history_values[backtest_start:backtest_start + 60].reshape(-1, 1)
                    scaled_backtest = metric_scaler.transform(backtest_input)
                    X_backtest = torch.tensor(scaled_backtest, dtype=torch.float32).unsqueeze(0)
                    
                    # Generate backtested predictions (what we would have predicted 24h ago)
                    backtest_preds = []
                    current_input = X_backtest
                    for _ in range(24):
                        with torch.no_grad():
                            pred = model(current_input)
                        backtest_preds.append(pred.item())
                        new_step = pred.unsqueeze(1)
                        current_input = torch.cat((current_input[:, 1:, :], new_step), dim=1)
                    
                    backtest_preds = np.array(backtest_preds).reshape(-1, 1)
                    backtest_inv = metric_scaler.inverse_transform(backtest_preds).flatten()
                    
                    # Get actual values for comparison (last 24 hours)
                    actual_24h = history_values[-24:]
                    actual_timestamps = history_timestamps[-24:]
                    
                    # Build comparison data (actual vs predicted for same timepoints)
                    comparison = []
                    for i in range(24):
                        comparison.append({
                            'timestamp': actual_timestamps[i],
                            'actual': float(actual_24h[i]),
                            'predicted': float(max(0, backtest_inv[i]))
                        })
                    
                    # --- FUTURE FORECAST: Predict next 24 hours ---
                    last_60 = history_values[-60:].reshape(-1, 1)
                    scaled_input = metric_scaler.transform(last_60)
                    X_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)
                    
                    forecasts = []
                    current_input = X_tensor
                    
                    for _ in range(24):
                        with torch.no_grad():
                            pred = model(current_input)
                        forecasts.append(pred.item())
                        new_step = pred.unsqueeze(1)
                        current_input = torch.cat((current_input[:, 1:, :], new_step), dim=1)
                    
                    forecasts = np.array(forecasts).reshape(-1, 1)
                    inv_forecasts = metric_scaler.inverse_transform(forecasts).flatten()
                    
                    # Prepare future forecast response
                    last_ts = history_timestamps[-1]
                    hour_ms = 3600000
                    
                    forecast_response = []
                    for val in inv_forecasts:
                        last_ts += hour_ms
                        # Round count-based metrics to integers
                        final_val = float(max(0, val))
                        if metric in ['Load', 'Error', 'Slow']:
                            final_val = int(round(final_val))
                        forecast_response.append({'timestamp': last_ts, 'value': final_val})
                    
                    # Calculate accuracy metrics
                    if np.mean(actual_24h) > 0:
                        # Round backtested predictions for counts too
                        predicted_backtest = backtest_inv
                        if metric in ['Load', 'Error', 'Slow']:
                            predicted_backtest = np.round(np.maximum(0, backtest_inv))
                        
                        mae = np.mean(np.abs(actual_24h - predicted_backtest))
                        den = np.mean(actual_24h)
                        mape = (mae / den) * 100 if den > 0 else 0
                    else:
                        mae = 0
                        mape = 0

                else: # ANN / GRU Autoencoder
                    # Anomaly Detection / Reconstruction View
                    last_60_vals = history_values[-60:]
                    last_60_ts = history_timestamps[-60:]
                    
                    last_60_reshaped = last_60_vals.reshape(-1, 1)
                    scaled_input = metric_scaler.transform(last_60_reshaped)
                    X_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0) # (1, 60, 1)
                    
                    with torch.no_grad():
                         reconstruction = model(X_tensor) # (1, 60, 1)
                    
                    reconstruction_np = reconstruction.squeeze(0).numpy()
                    reconstructed_vals = metric_scaler.inverse_transform(reconstruction_np).flatten()
                    
                    comparison = []
                    # Last 24 points are indices -24 to end
                    for i in range(36, 60): # 60 - 24 = 36
                         comparison.append({
                             'timestamp': last_60_ts[i],
                             'actual': float(last_60_vals[i]),
                             'predicted': float(max(0, reconstructed_vals[i]))
                         })
                    
                    # Error metric: Reconstruction Error (MAE)
                    diff = np.abs(last_60_vals[-24:] - reconstructed_vals[-24:])
                    mae = np.mean(diff)
                    den = np.mean(last_60_vals[-24:])
                    mape = (mae / den) * 100 if den > 0 else 0
                    
                    forecast_response = []
                
                return jsonify({
                    'comparison': comparison,
                    'forecast': forecast_response,
                    'model': model_type,
                    'accuracy': {
                        'mae': float(mae),
                        'mape': float(mape)
                    }
                })
                
                # Update comparison data with rounded values
                comparison = []
                for i in range(24):
                    pred_val = float(max(0, backtest_inv[i]))
                    if metric in ['Load', 'Error', 'Slow']:
                        pred_val = int(round(pred_val))
                        
                    comparison.append({
                        'timestamp': actual_timestamps[i],
                        'actual': float(actual_24h[i]),
                        'predicted': pred_val
                    })

                return jsonify({
                    'comparison': comparison,  # Last 24h: actual vs predicted
                    'forecast': forecast_response,  # Next 24h predictions
                    'model': 'LSTM',
                    'accuracy': {
                        'mae': float(mae),
                        'mape': float(min(mape, 100))  # Cap error at 100%
                    }
                })
            except Exception as e:
                print(f"LSTM prediction failed for {metric}, falling back to statistical: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback: Statistical forecast (moving average + trend)
        window = min(24, len(history_values))
        recent_values = history_values[-window:]
        
        # Calculate trend
        if len(recent_values) >= 2:
            trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
        else:
            trend = 0
        
        # Calculate weighted moving average (more weight to recent values)
        weights = np.exp(np.linspace(0, 1, len(recent_values)))
        weights /= weights.sum()
        weighted_avg = np.average(recent_values, weights=weights)
        
        # Calculate hourly pattern (hour of day effect)
        hourly_pattern = np.zeros(24)
        hourly_counts = np.zeros(24)
        for i, val in enumerate(history_values):
            hour_of_day = (hourly.index[i].hour) % 24
            hourly_pattern[hour_of_day] += val
            hourly_counts[hour_of_day] += 1
        
        # Avoid division by zero
        hourly_counts = np.where(hourly_counts == 0, 1, hourly_counts)
        hourly_avg = hourly_pattern / hourly_counts
        overall_avg = np.mean(hourly_avg) if np.mean(hourly_avg) > 0 else 1
        hourly_factors = hourly_avg / overall_avg
        
        # Generate comparison (last 24h actual vs simple prediction)
        comparison = []
        for i in range(min(24, len(history_values))):
            idx = len(history_values) - 24 + i
            if idx >= 0:
                comparison.append({
                    'timestamp': history_timestamps[idx],
                    'actual': float(history_values[idx]),
                    'predicted': float(weighted_avg * hourly_factors[hourly.index[idx].hour])
                })
        
        # Generate future forecast
        last_ts = history_timestamps[-1]
        last_hour = hourly.index[-1].hour
        hour_ms = 3600000
        
        forecast_response = []
        current_value = weighted_avg
        
        for i in range(24):
            forecast_hour = (last_hour + i + 1) % 24
            hourly_factor = hourly_factors[forecast_hour] if hourly_factors[forecast_hour] > 0 else 1
            
            predicted_value = current_value * hourly_factor + trend * (i + 1)
            predicted_value = max(0, predicted_value)
            
            last_ts += hour_ms
            forecast_response.append({'timestamp': last_ts, 'value': float(predicted_value)})
            
            current_value = 0.9 * current_value + 0.1 * weighted_avg
        
        return jsonify({
            'comparison': comparison,
            'forecast': forecast_response,
            'model': 'Statistical',
            'accuracy': {'mae': 0, 'mape': 0}
        })
        
    except Exception as e:
        print(f"Forecast error for {metric}: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Calculate duration in minutes based on timeframe as fallback
    duration_map = {
        '1h': 60, '6h': 360, '24h': 1440, '7d': 10080, '30d': 43200,
        '6m': 259200, '1y': 525600, 'all': 43200, '5m': 30, '15m': 30
    }
    default_duration = duration_map.get(timeframe, 43200)
    
    # Get official time parameters (reads start_date/end_date)
    duration, start_time_ms, end_time_ms = get_time_params(default_duration=default_duration)
    
    # Get URL and fetch data
    url = get_bt_url(tier, bt, 'Error')
    if not url:
        return jsonify({'error': f'No URL found for tier={tier}, bt={bt}, metric_type=Error', 'total': 0})
    
    raw_data = fetch_from_appdynamics(url, duration, start_time=start_time_ms, end_time=end_time_ms)
    
    if not raw_data:
        return jsonify({'error': 'Failed to fetch data from AppDynamics', 'total': 0})
    
    try:
        if not raw_data:
            return jsonify({'error': 'No data returned', 'total': 0})
            
        # Data is already flattened by fetch_from_appdynamics
        all_values = raw_data
            
        df = pd.DataFrame(all_values)
        if df.empty:
            return jsonify({'error': 'Empty dataframe', 'total': 0})
        
        # Preprocessing
        df['dt'] = pd.to_datetime(df['startTimeInMillis'], unit='ms') + pd.Timedelta(hours=7)
        
        df['hour'] = df['dt'].dt.hour
        df['day_name'] = df['dt'].dt.day_name()
        
        # Aggregation - use 'sum' column if available, else 'value'
        value_col = 'sum' if 'sum' in df.columns else 'value'
        
        # Calculate total errors
        total_errors = int(df[value_col].sum())
        


        if timeframe == '6h':
            # Create a full 6-hour range ending now
            end_time = pd.Timestamp.now().floor('min')
            start_time = end_time - pd.Timedelta(hours=6) + pd.Timedelta(minutes=1) # last 360 mins inclusive
            full_range = pd.date_range(start=start_time, end=end_time, freq='min')
            
            # Resample strictly to this range
            df_resampled = df.set_index('dt').resample('min')[value_col].sum().reindex(full_range, fill_value=0)
            
            # Create helper columns for pivoting
            df_pivot = df_resampled.reset_index()
            df_pivot.columns = ['dt', 'val']
            df_pivot['hour_label'] = df_pivot['dt'].dt.strftime('%H:00')
            df_pivot['minute_col'] = df_pivot['dt'].dt.minute
            
            # Heatmap: Rows = Hours (e.g. 10:00, 11:00), Cols = Minutes (0-59)
            heatmap_pivot = df_pivot.pivot(index='hour_label', columns='minute_col', values='val').fillna(0)
            # Ensure all 60 minutes are present as columns
            heatmap_pivot = heatmap_pivot.reindex(columns=range(60), fill_value=0)
            
            # Recalculate total
            total_errors = int(df_resampled.sum())
            
            hourly_dist = df_resampled.tolist() # Flattened list still useful for hourly chart if needed
            daily_dist = [total_errors]
            peak_hour = f"Minute {np.argmax(hourly_dist)}"
            peak_day = "Today"

        elif timeframe == '1h':
            # Minute-level aggregation for last hour
            minutes_count = 60
            
            # Anchor to NOW
            end_time = pd.Timestamp.now().floor('min')
            start_time = end_time - pd.Timedelta(minutes=minutes_count - 1)
            full_range = pd.date_range(start=start_time, end=end_time, freq='min')
            
            df_resampled = df.set_index('dt').resample('min')[value_col].sum().reindex(full_range, fill_value=0)
            
            heatmap_pivot = df_resampled.to_frame().T
            heatmap_pivot.columns = heatmap_pivot.columns.strftime('%H:%M')
            heatmap_pivot.index = ['Last Hour']
            
            # Recalculate total
            total_errors = int(df_resampled.sum())
            
            hourly_dist = df_resampled.tolist()
            daily_dist = [total_errors]
            peak_hour = f"Minute {np.argmax(hourly_dist)}"
            peak_day = "Today"

        elif timeframe in ['5m', '15m']:
            # Linear minute aggregation for short durations
            # Create a full range of timestamps for the last N minutes
            minutes_count = 5 if timeframe == '5m' else 15
            
            # Anchor to NOW, not the data's max time.
            # This ensures we show the actual "Last 5 Minutes" even if data is old/missing.
            end_time = pd.Timestamp.now().floor('min')
            start_time = end_time - pd.Timedelta(minutes=minutes_count - 1)
            
            full_range = pd.date_range(start=start_time, end=end_time, freq='min')
            
            # Resample and fill missing with 0
            df_resampled = df.set_index('dt').resample('min')[value_col].sum().reindex(full_range, fill_value=0)
            
            # Heatmap: 1 row, N columns
            # Heatmap: 1 row, N columns
            heatmap_pivot = df_resampled.to_frame().T
            heatmap_pivot.columns = heatmap_pivot.columns.strftime('%H:%M')
            heatmap_pivot.index = [f'Last {minutes_count} Mins']
            
            # Recalculate total for this short timeframe to match the heatmap
            total_errors = int(df_resampled.sum())
            
            # "Hourly" dist -> actually minute dist here
            hourly_dist = df_resampled.tolist()
            daily_dist = [total_errors]
            peak_hour = f"Minute {np.argmax(hourly_dist)}"
            peak_day = "Today"

        elif timeframe == '24h':
            # Hourly aggregation for last 24h, specific dates as rows
            df['date_short'] = df['dt'].dt.strftime('%d-%b')
            
            # Heatmap: Rows = Dates, Cols = Hours
            heatmap_pivot = df.groupby(['date_short', 'hour'])[value_col].sum().reset_index().pivot(index='date_short', columns='hour', values=value_col).fillna(0)
            heatmap_pivot = heatmap_pivot.reindex(columns=range(24), fill_value=0).fillna(0)
            
            # Standard hourly dist
            hourly_dist = df.groupby('hour')[value_col].sum().reindex(range(24), fill_value=0).tolist()
            
            daily_dist = [total_errors]
            
            peak_hour_idx = np.argmax(hourly_dist)
            peak_hour = f"{peak_hour_idx:02d}:00"
            peak_day = "Today"
            
        else:
            # Standard Day vs Hour
            heatmap_pivot = df.groupby(['day_name', 'hour'])[value_col].sum().reset_index().pivot(index='day_name', columns='hour', values=value_col).fillna(0)
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_pivot = heatmap_pivot.reindex(days_order).reindex(columns=range(24), fill_value=0).fillna(0)
            
            hourly_dist = df.groupby('hour')[value_col].sum().reindex(range(24), fill_value=0).tolist()
            daily_dist = df.groupby('day_name')[value_col].sum().reindex(days_order, fill_value=0).tolist()
            
            peak_hour_idx = np.argmax(hourly_dist)
            peak_hour = f"{peak_hour_idx:02d}:00"
            peak_day = days_order[np.argmax(daily_dist)] if daily_dist else 'N/A'
        
        return jsonify({
            'heatmap': heatmap_pivot.to_dict(orient='split'),
            'hourly': hourly_dist,
            'daily': daily_dist,
            'total': total_errors,
            'peak_hour': peak_hour,
            'peak_day': peak_day,
            'min': int(df[df[value_col] > 0][value_col].min()) if not df[df[value_col] > 0].empty else 0,
            'max': int(df[value_col].max())
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
    
    # Calculate duration in minutes based on timeframe as fallback
    duration_map = {
        '1h': 60, '6h': 360, '24h': 1440, '7d': 10080, '30d': 43200,
        '6m': 259200, '1y': 525600, 'all': 43200, '5m': 30, '15m': 30
    }
    default_duration = duration_map.get(timeframe, 43200)

    # Get official time parameters (reads start_date/end_date)
    duration, start_time_ms, end_time_ms = get_time_params(default_duration=default_duration)
    
    # Get URL and fetch data
    url = get_bt_url(tier, bt, 'Load')
    if not url:
        return jsonify({'error': f'No URL found for tier={tier}, bt={bt}, metric_type=Load', 'total': 0})
    
    raw_data = fetch_from_appdynamics(url, duration, start_time=start_time_ms, end_time=end_time_ms)
    
    if not raw_data:
        return jsonify({'error': 'Failed to fetch data from AppDynamics', 'total': 0})
    
    try:
        if not raw_data:
            return jsonify({'error': 'No data returned', 'total': 0})
            
        # Data is already flattened by fetch_from_appdynamics
        all_values = raw_data
            
        df = pd.DataFrame(all_values)
        if df.empty:
            return jsonify({'error': 'Empty dataframe', 'total': 0})

        # Preprocessing
        df['dt'] = pd.to_datetime(df['startTimeInMillis'], unit='ms') + pd.Timedelta(hours=7)
        
        df['hour'] = df['dt'].dt.hour
        df['day_name'] = df['dt'].dt.day_name()
        
        # Aggregation - use 'sum' column if available, else 'value'
        value_col = 'sum' if 'sum' in df.columns else 'value'
        
        # Calculate total load
        total_load = int(df[value_col].sum())
        
        if timeframe == '6h':
            # Create a full 6-hour range ending now
            end_time = pd.Timestamp.now().floor('min')
            start_time = end_time - pd.Timedelta(hours=6) + pd.Timedelta(minutes=1)
            full_range = pd.date_range(start=start_time, end=end_time, freq='min')
            
            # Resample strictly to this range
            df_resampled = df.set_index('dt').resample('min')[value_col].sum().reindex(full_range, fill_value=0)
            
            # Create helper columns for pivoting
            df_pivot = df_resampled.reset_index()
            df_pivot.columns = ['dt', 'val']
            df_pivot['hour_label'] = df_pivot['dt'].dt.strftime('%H:00')
            df_pivot['minute_col'] = df_pivot['dt'].dt.minute
            
            # Heatmap: Rows = Hours, Cols = Minutes
            heatmap_pivot = df_pivot.pivot(index='hour_label', columns='minute_col', values='val').fillna(0)
            heatmap_pivot = heatmap_pivot.reindex(columns=range(60), fill_value=0)
            
            # Recalculate total
            total_load = int(df_resampled.sum())
            
            hourly_dist = df_resampled.tolist()
            daily_dist = [total_load]
            peak_hour = f"Minute {np.argmax(hourly_dist)}"
            peak_day = "Today"

        elif timeframe == '1h':
            # Minute-level aggregation for last hour
            minutes_count = 60
            
            # Anchor to NOW
            end_time = pd.Timestamp.now().floor('min')
            start_time = end_time - pd.Timedelta(minutes=minutes_count - 1)
            full_range = pd.date_range(start=start_time, end=end_time, freq='min')
            
            df_resampled = df.set_index('dt').resample('min')[value_col].sum().reindex(full_range, fill_value=0)
            
            heatmap_pivot = df_resampled.to_frame().T
            heatmap_pivot.columns = heatmap_pivot.columns.strftime('%H:%M')
            heatmap_pivot.index = ['Last Hour']
            
            # Recalculate total
            total_load = int(df_resampled.sum())
            
            hourly_dist = df_resampled.tolist()
            daily_dist = [total_load]
            peak_hour = f"Minute {np.argmax(hourly_dist)}"
            peak_day = "Today"

        elif timeframe in ['5m', '15m']:
            # Linear minute aggregation
            minutes_count = 5 if timeframe == '5m' else 15
            
            # Fix: Anchor to NOW
            end_time = pd.Timestamp.now().floor('min')
            start_time = end_time - pd.Timedelta(minutes=minutes_count - 1)
            full_range = pd.date_range(start=start_time, end=end_time, freq='min')
            
            df_resampled = df.set_index('dt').resample('min')[value_col].sum().reindex(full_range, fill_value=0)
            
            heatmap_pivot = df_resampled.to_frame().T
            # Fix: Format columns as HH:mm
            heatmap_pivot.columns = heatmap_pivot.columns.strftime('%H:%M')
            heatmap_pivot.index = [f'Last {minutes_count} Mins']
            
            # Recalculate total for this short timeframe
            total_load = int(df_resampled.sum())
            
            hourly_dist = df_resampled.tolist()
            daily_dist = [total_load]
            peak_hour = f"Minute {np.argmax(hourly_dist)}"
            peak_day = "Today"

        elif timeframe == '24h':
            # Hourly aggregation for last 24h, specific dates as rows
            df['date_short'] = df['dt'].dt.strftime('%d-%b')
            
            # Heatmap: Rows = Dates, Cols = Hours
            heatmap_pivot = df.groupby(['date_short', 'hour'])[value_col].sum().reset_index().pivot(index='date_short', columns='hour', values=value_col).fillna(0)
            heatmap_pivot = heatmap_pivot.reindex(columns=range(24), fill_value=0).fillna(0)
            
            # Standard hourly dist
            hourly_dist = df.groupby('hour')[value_col].sum().reindex(range(24), fill_value=0).tolist()
            
            daily_dist = [total_load]
            
            peak_hour_idx = np.argmax(hourly_dist)
            peak_hour = f"{peak_hour_idx:02d}:00"
            peak_day = "Today"
            
        else:
            # Standard Day vs Hour
            heatmap_pivot = df.groupby(['day_name', 'hour'])[value_col].sum().reset_index().pivot(index='day_name', columns='hour', values=value_col).fillna(0)
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_pivot = heatmap_pivot.reindex(days_order).reindex(columns=range(24), fill_value=0).fillna(0)
            
            hourly_dist = df.groupby('hour')[value_col].sum().reindex(range(24), fill_value=0).tolist()
            daily_dist = df.groupby('day_name')[value_col].sum().reindex(days_order, fill_value=0).tolist()
            
            peak_hour = f"{np.argmax(hourly_dist)}:00"
            peak_day = days_order[np.argmax(daily_dist)] if daily_dist else 'N/A'
        
        return jsonify({
            'heatmap': heatmap_pivot.to_dict(orient='split'),
            'hourly': hourly_dist,
            'daily': daily_dist,
            'total': total_load,
            'peak_hour': peak_hour,
            'peak_day': peak_day,
            'min': int(df[df[value_col] > 0][value_col].min()) if not df[df[value_col] > 0].empty else 0,
            'max': int(df[value_col].max())
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
    
    # Calculate duration in minutes based on timeframe as fallback
    duration_map = {
        '1h': 60, '6h': 360, '24h': 1440, '7d': 10080, '30d': 43200,
        '6m': 259200, '1y': 525600, 'all': 43200, '5m': 30, '15m': 30
    }
    default_duration = duration_map.get(timeframe, 43200)

    # Get official time parameters (reads start_date/end_date)
    duration, start_time_ms, end_time_ms = get_time_params(default_duration=default_duration)
    
    # Map metric type to CSV metric name
    metric_name = 'Very Slow' if metric_type == 'veryslow' else 'Slow'
    
    # Get URL and fetch data
    url = get_bt_url(tier, bt, metric_name)
    if not url:
        return jsonify({'error': f'No URL found for tier={tier}, bt={bt}, metric_type={metric_name}', 'total': 0})
    
    raw_data = fetch_from_appdynamics(url, duration, start_time=start_time_ms, end_time=end_time_ms)
    
    if not raw_data:
        return jsonify({'error': 'Failed to fetch data from AppDynamics', 'total': 0})
    
    try:
        if not raw_data:
            return jsonify({'error': 'No data returned', 'total': 0})
            
        # Data is already flattened by fetch_from_appdynamics
        all_values = raw_data
            
        df = pd.DataFrame(all_values)
        if df.empty:
            return jsonify({'error': 'Empty dataframe', 'total': 0})

        # Preprocessing
        df['dt'] = pd.to_datetime(df['startTimeInMillis'], unit='ms') + pd.Timedelta(hours=7)
        
        df['hour'] = df['dt'].dt.hour
        df['day_name'] = df['dt'].dt.day_name()
        
        # Aggregation - use 'sum' column if available, else 'value'
        value_col = 'sum' if 'sum' in df.columns else 'value'
        
        # Calculate total slow calls (default)
        total_slow = int(df[value_col].sum())
        
        if timeframe == '6h':
            # Create a full 6-hour range ending now
            end_time = pd.Timestamp.now().floor('min')
            start_time = end_time - pd.Timedelta(hours=6) + pd.Timedelta(minutes=1)
            full_range = pd.date_range(start=start_time, end=end_time, freq='min')
            
            # Resample strictly to this range
            df_resampled = df.set_index('dt').resample('min')[value_col].sum().reindex(full_range, fill_value=0)
            
            # Create helper columns for pivoting
            df_pivot = df_resampled.reset_index()
            df_pivot.columns = ['dt', 'val']
            df_pivot['hour_label'] = df_pivot['dt'].dt.strftime('%H:00')
            df_pivot['minute_col'] = df_pivot['dt'].dt.minute
            
            # Heatmap: Rows = Hours, Cols = Minutes
            heatmap_pivot = df_pivot.pivot(index='hour_label', columns='minute_col', values='val').fillna(0)
            heatmap_pivot = heatmap_pivot.reindex(columns=range(60), fill_value=0)
            
            # Recalculate total
            total_slow = int(df_resampled.sum())
            
            hourly_dist = df_resampled.tolist()
            daily_dist = [total_slow]
            peak_hour = f"Minute {np.argmax(hourly_dist)}"
            peak_day = "Today"

        elif timeframe == '1h':
            # Minute-level aggregation for last hour
            minutes_count = 60
            
            # Anchor to NOW
            end_time = pd.Timestamp.now().floor('min')
            start_time = end_time - pd.Timedelta(minutes=minutes_count - 1)
            full_range = pd.date_range(start=start_time, end=end_time, freq='min')
            
            df_resampled = df.set_index('dt').resample('min')[value_col].sum().reindex(full_range, fill_value=0)
            
            heatmap_pivot = df_resampled.to_frame().T
            heatmap_pivot.columns = heatmap_pivot.columns.strftime('%H:%M')
            heatmap_pivot.index = ['Last Hour']
            
            # Recalculate total
            total_slow = int(df_resampled.sum())
            
            hourly_dist = df_resampled.tolist()
            daily_dist = [total_slow]
            peak_hour = f"Minute {np.argmax(hourly_dist)}"
            peak_day = "Today"

        elif timeframe in ['5m', '15m']:
            # Linear minute aggregation
            minutes_count = 5 if timeframe == '5m' else 15
            
            # Anchor to NOW
            end_time = pd.Timestamp.now().floor('min')
            start_time = end_time - pd.Timedelta(minutes=minutes_count - 1)
            full_range = pd.date_range(start=start_time, end=end_time, freq='min')
            
            df_resampled = df.set_index('dt').resample('min')[value_col].sum().reindex(full_range, fill_value=0)
            
            heatmap_pivot = df_resampled.to_frame().T
            # Fix: Format columns as HH:mm
            heatmap_pivot.columns = heatmap_pivot.columns.strftime('%H:%M')
            heatmap_pivot.index = [f'Last {minutes_count} Mins']
            
            # Recalculate total for this short timeframe
            total_slow = int(df_resampled.sum())
            
            hourly_dist = df_resampled.tolist()
            daily_dist = [total_slow]
            peak_hour = f"Minute {np.argmax(hourly_dist)}"
            peak_day = "Today"

        elif timeframe == '24h':
            # Hourly aggregation for last 24h, specific dates as rows
            df['date_short'] = df['dt'].dt.strftime('%d-%b')
            
            # Heatmap: Rows = Dates, Cols = Hours
            heatmap_pivot = df.groupby(['date_short', 'hour'])[value_col].sum().reset_index().pivot(index='date_short', columns='hour', values=value_col).fillna(0)
            heatmap_pivot = heatmap_pivot.reindex(columns=range(24), fill_value=0).fillna(0)
            
            # Standard hourly dist
            hourly_dist = df.groupby('hour')[value_col].sum().reindex(range(24), fill_value=0).tolist()
            daily_dist = [total_slow]
            
            peak_hour_idx = np.argmax(hourly_dist)
            peak_hour = f"{peak_hour_idx:02d}:00"
            peak_day = "Today"
            
        else:
            # Standard Day vs Hour
            heatmap_pivot = df.groupby(['day_name', 'hour'])[value_col].sum().reset_index().pivot(index='day_name', columns='hour', values=value_col).fillna(0)
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_pivot = heatmap_pivot.reindex(days_order).reindex(columns=range(24), fill_value=0).fillna(0)
            
            hourly_dist = df.groupby('hour')[value_col].sum().reindex(range(24), fill_value=0).tolist()
            daily_dist = df.groupby('day_name')[value_col].sum().reindex(days_order, fill_value=0).tolist()
            
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
        off_hours_calls = total_slow - business_calls
        
        impact_data = {
            'Business Hours (8-18)': business_calls,
            'Off-Hours': off_hours_calls
        }
        
        return jsonify({
            'heatmap': heatmap_pivot.to_dict(orient='split'),
            'hourly': hourly_dist,
            'daily': daily_dist,
            'total': total_slow,
            'peak_hour': peak_hour,
            'peak_day': peak_day,
            'trend': trend_data,
            'impact': impact_data,
            'min': int(df[df[value_col] > 0][value_col].min()) if not df[df[value_col] > 0].empty else 0,
            'max': int(df[value_col].max()) if not df.empty else 0
        })
        
    except Exception as e:
        print(f"Slow Calls Analysis Error: {e}")
        return jsonify({'error': str(e)})

@app.route('/jvm-health')
def jvm_health_page():
    return render_template('jvm_health.html')

@app.route('/api/jvm-data')
def get_jvm_data():
    # Get time params using 60 mins default
    duration, start_time_ms, end_time_ms = get_time_params(default_duration=60)
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
    
    # Helper to fetch data using central logic
    def fetch_metric_flexible(metric_path):
        target_app = "Smart2"
        encoded_metric = urllib.parse.quote(metric_path)
        base_url = f"{BASE_URL}/controller/rest/applications/{target_app}/metric-data?metric-path={encoded_metric}"
        
        return fetch_from_appdynamics(base_url, duration, start_time=start_time_ms, end_time=end_time_ms)

    # Fetch all metrics
    for key, path in METRICS.items():
        data = fetch_metric_flexible(path)
        
        # Process data into simple Time/Value pairs
        series = []
        if data and isinstance(data, list):
            for val in data:
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
        
        # 0. Recalculate Health based on new thresholds (to match Frontend Legend)
        # Normal: < 1000ms, < 1% Errors
        # Warning: 1000ms-5000ms, 1-10% Errors
        # Critical: > 5000ms, > 10% Errors
        def calculate_health(row):
            t = row['Response Time (ms)']
            e = row['% Errors']
            if t > 5000 or e > 10:
                return 'Critical'
            if t >= 1000 or e >= 1:
                return 'Warning'
            return 'Normal'
        
        df['Health'] = df.apply(calculate_health, axis=1)
        
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
        df['Health'] = df['Health'].fillna('Unknown')
        scatter_data = df[['Name', 'Calls', 'Response Time (ms)', 'Health']].to_dict(orient='records')

        # 6. Table Data
        table_cols = ['Name', 'Health', 'Response Time (ms)', 'Calls', '% Errors']
        if 'Tier' in df.columns:
            table_cols.append('Tier')
        table_data = df[table_cols].to_dict(orient='records')

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
        print(f"Error processing stats: {e}")
        return jsonify({'error': str(e)})

@app.route('/database-analysis')
def database_analysis_page():
    return render_template('database_analysis.html')

@app.route('/api/database-analysis')
def get_database_analysis():
    # 1. Fetch Data
    duration, start_time_ms, end_time_ms = get_time_params(default_duration=60)

    # Metric Paths
    metric_time_spent = "Databases|C2L_DMS|KPI|Time Spent in Executions (s)"
    metric_load = "Databases|C2L_DMS|KPI|Calls per Minute"
    metric_cpu = "Databases|C2L_DMS|Hardware Resources|CPU|%Busy"
    
    # Helper to fetch data
    def fetch_metric(m_path):
        target_app = "Database Monitoring"
        encoded_metric = urllib.parse.quote(m_path)
        base_url = f"{BASE_URL}/controller/rest/applications/{urllib.parse.quote(target_app)}/metric-data?metric-path={encoded_metric}"
        
        # Use our updated central function
        return fetch_from_appdynamics(base_url, duration, start_time=start_time_ms, end_time=end_time_ms)

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
            
    # Spike Detection (Statistical: > Mean + 2*StdDev)
    import numpy as np
    spike_count = 0
    if values_time:
        mean_time = np.mean(values_time)
        std_time = np.std(values_time)
        threshold = mean_time + (2 * std_time)
        # minimal threshold to avoid noise in low-latency environments
        threshold = max(threshold, 0.5) 
         
        spike_count = sum(1 for v in values_time if v > threshold)

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
        'spike_count': spike_count,
        'total_points': total_points,
        'queries': queries_data
    })


# Background Monitor Global State
last_alert_minutes = {} # Key: (tier, bt), Value: timestamp

def background_error_monitor():
    """
    Runs in the background to check for errors every minute.
    Iterates through ALL Business Transactions defined in the CSV with MetricType='Error'.
    Alerts only if errors are found in the LATEST minute to avoid duplicates.
    """
    global last_alert_minutes
    global last_alert_minutes
    monitor_logger.info("Background Error Monitor Started (Checking All BTs every 5 minutes)...")
    
    while True:
        try:
            # 1. Load BTs
            df_bts = load_bt_csv()
            if df_bts.empty:
                monitor_logger.warning("BT CSV empty or not found. Skipping monitor cycle.")
                time.sleep(60) # Retry sooner if file missing
                continue
                
            # Filter for Error metric type
            error_bts = df_bts[df_bts['MetricType'] == 'Error']
            monitor_logger.info(f"Starting Monitor Cycle: Checking {len(error_bts)} Business Transactions...")
            
            # 2. Iterate through each BT
            checks_count = 0
            alerts_count = 0
            
            for index, row in error_bts.iterrows():
                tier = row['Tier']
                bt = row['BT']
                url = row['URL']
                
                if not url: continue
                
                try:
                    # Use the new fetch_from_appdynamics helper
                    # Fetch last 20 minutes to ensure we don't miss any data points between 5-min cycles
                    data = fetch_from_appdynamics(url, duration=20)
                    checks_count += 1
                    
                    if not data: 
                        monitor_logger.warning(f"[Empty Response] BT: {bt} | Tier: {tier}")
                        continue
        
                    df = pd.DataFrame(data)
                    if df.empty: 
                        monitor_logger.warning(f"[Empty DataFrame] BT: {bt} | Tier: {tier}")
                        continue
                    
                    # Process timestamps
                    df['dt'] = pd.to_datetime(df['startTimeInMillis'], unit='ms') + pd.Timedelta(hours=7)
                    
                    # Sort by time ascending so we process oldest-to-newest
                    df.sort_values('dt', inplace=True)
                    
                    # Unique key for this BT
                    bt_key = (tier, bt)
                    
                    # Get last alerted timestamp for this BT
                    last_alert_ts = last_alert_minutes.get(bt_key)

                    # Iterate through ALL 20 minutes of data points
                    for _, row in df.iterrows():
                        row_dt = row['dt']
                        
                        # Use 'sum' if available, else 'value'
                        val_col = 'sum' if 'sum' in df.columns else 'value'
                        val = row[val_col]
                        
                        # Logic:
                        # 1. Must be an error (value > 0)
                        # 2. Must be NEWER than the last alert we sent (to avoid duplicates)
                        # 3. If last_alert_ts is None (first run), we might want to alert, 
                        #    BUT to avoid startup spam of old errors, maybe we rely on the loop catching up?
                        #    Let's alert if val > 0 and (never alerted OR new timestamp).
                        
                        if val > 0:
                            if last_alert_ts is None or row_dt > last_alert_ts:
                                # FOUND NEW ERROR
                                monitor_logger.info(f"[Success] BT: {bt} | Time: {row_dt.strftime('%H:%M')} | Errors: {int(val)}")

                                alert_msg = (
                                    f"⚠️ *Background Monitor Alert*\n"
                                    f"New Errors Detected: `{int(val)}`\n"
                                    f"Time: `{row_dt.strftime('%H:%M')}`\n"
                                    f"BT: `{bt}`\n"
                                    f"Tier: `{tier}`"
                                )
                                send_telegram_alert(alert_msg)
                                monitor_logger.warning(f"ALERT SENT for {bt}: {int(val)} errors at {row_dt.strftime('%H:%M')}")
                                
                                # Update last alert to THIS timestamp
                                last_alert_minutes[bt_key] = row_dt
                                last_alert_ts = row_dt # Update local var for next iteration loop
                                alerts_count += 1
                            else:
                                # Already alerted this timestamp, skip
                                pass
                        else:
                            # logging success for 0 errors? verify user request
                            # User asked for "Success" proof. We can log the latest one outside loop?
                            pass
                            
                    # Log heartbeat for latest point (even if 0 errors) to show it's working
                    latest_dt = df['dt'].max()
                    latest_val = df[df['dt'] == latest_dt][val_col].sum()
                    monitor_logger.info(f"[Heartbeat] {bt} checked up to {latest_dt.strftime('%H:%M')} (Latest Value: {int(latest_val)})")

                    # Small sleep to avoid hammering the API too fast in the loop
                    time.sleep(0.5) 
                    
                except Exception as inner_e:
                    monitor_logger.error(f"Error checking BT {bt}: {inner_e}")
                    continue
            
            monitor_logger.info(f"Cycle Completed. Checked {checks_count} BTs. Sent {alerts_count} Alerts. Waiting 5 minutes...")
            
            # wait 300 seconds (5 minutes) between full check cycles
            time.sleep(300)

        except Exception as e:
            monitor_logger.error(f"Background monitor critical error: {e}")
            time.sleep(60) # Wait before retry on crash
            time.sleep(60) # wait before retrying

if __name__ == '__main__':
    # Ensure thread starts only once (Flask reloader protection)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        monitor_thread = threading.Thread(target=background_error_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
    app.run(debug=True, port=5000)
