from flask import Flask, render_template, jsonify, request
import json
import os
from collections import Counter
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
    return render_template('dashboard.html')

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
