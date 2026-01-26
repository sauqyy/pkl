"""
LSTM Time Series Forecasting Model
===================================
A complete PyTorch LSTM model for forecasting time series metrics.
Supports: Load, Response Time, Errors, Slow Calls (or any numerical time series)

Usage:
    1. Training:   python forecast_model.py train --data your_data.json
    2. Prediction: python forecast_model.py predict --model lstm_model.pth --hours 24
    3. Demo:       python forecast_model.py demo
"""

import json
import numpy as np
import pandas as pd
import joblib
import os
import argparse
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'LOOK_BACK': 60,        # Number of past hours to use for prediction
    'HIDDEN_SIZE': 50,      # LSTM hidden layer size
    'NUM_LAYERS': 2,        # Number of LSTM layers
    'OUTPUT_SIZE': 1,       # Predicting 1 value at a time
    'NUM_EPOCHS': 100,      # Training epochs
    'BATCH_SIZE': 16,       # Batch size for training
    'LEARNING_RATE': 0.001, # Adam optimizer learning rate
    'FORECAST_HOURS': 24,   # Default hours to forecast
}

# ============================================================================
# LSTM Model Definition
# ============================================================================
class LSTMModel(nn.Module):
    """
    LSTM Neural Network for Time Series Forecasting.
    
    Architecture:
        Input -> LSTM (multi-layer) -> Fully Connected -> Output
    
    Args:
        input_size: Number of features per timestep (1 for univariate)
        hidden_size: Number of LSTM hidden units
        num_layers: Number of stacked LSTM layers
        output_size: Number of output predictions (1 for single-step)
    """
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take only the last timestep output
        out = self.fc(out[:, -1, :])
        return out

# ============================================================================
# Dataset Class
# ============================================================================
class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# ============================================================================
# Data Preparation Functions
# ============================================================================
def create_sequences(data, look_back):
    """
    Create input/output sequences for LSTM training.
    
    Args:
        data: Normalized time series data (numpy array, shape: [n, 1])
        look_back: Number of past timesteps to use as input
    
    Returns:
        X: Input sequences (shape: [samples, look_back, 1])
        y: Target values (shape: [samples])
    """
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        sequence = data[i:(i + look_back), 0]
        target = data[i + look_back, 0]
        X.append(sequence)
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y


def load_data_from_json(filepath):
    """
    Load time series data from JSON file.
    
    Expected format:
        [{"startTimeInMillis": 1234567890000, "value": 123.45}, ...]
    
    Returns:
        DataFrame with 'timestamp' and 'value' columns
    """
    with open(filepath, 'r') as f:
        raw_data = json.load(f)
    
    df = pd.DataFrame(raw_data)
    
    # Handle different column names
    if 'startTimeInMillis' in df.columns:
        df['timestamp'] = pd.to_datetime(df['startTimeInMillis'], unit='ms')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values('timestamp')
    return df


def resample_to_hourly(df, value_column='value', agg_method='mean'):
    """
    Resample data to hourly intervals.
    
    Args:
        df: DataFrame with 'timestamp' and value column
        value_column: Name of the value column
        agg_method: 'mean' for response time, 'sum' for counts
    
    Returns:
        DataFrame with hourly data
    """
    df = df.set_index('timestamp')
    
    if agg_method == 'sum':
        hourly = df[value_column].resample('h').sum()
    else:
        hourly = df[value_column].resample('h').mean()
    
    hourly = hourly.dropna()
    return hourly.reset_index()

# ============================================================================
# Training Function
# ============================================================================
def train_model(data_path, model_path='lstm_model.pth', scaler_path='scaler.pkl', 
                config=None):
    """
    Train the LSTM model on time series data.
    
    Args:
        data_path: Path to JSON data file
        model_path: Path to save trained model
        scaler_path: Path to save fitted scaler
        config: Configuration dict (uses defaults if None)
    
    Returns:
        Trained model and scaler
    """
    if config is None:
        config = CONFIG
    
    print("=" * 60)
    print("LSTM Time Series Forecasting - Training")
    print("=" * 60)
    
    # Load data
    print(f"\n[1/5] Loading data from {data_path}...")
    df = load_data_from_json(data_path)
    print(f"      Loaded {len(df)} data points")
    
    # Prepare data
    print(f"\n[2/5] Preparing data...")
    data = df[['value']].values.astype('float32')
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    print(f"      Data range: {data.min():.2f} - {data.max():.2f}")
    
    # Create sequences
    print(f"\n[3/5] Creating sequences (look_back={config['LOOK_BACK']})...")
    X, y = create_sequences(scaled_data, config['LOOK_BACK'])
    print(f"      Created {len(X)} training sequences")
    
    # Create DataLoader
    dataset = TimeSeriesDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
    
    # Initialize model
    print(f"\n[4/5] Initializing LSTM model...")
    model = LSTMModel(
        input_size=1,
        hidden_size=config['HIDDEN_SIZE'],
        num_layers=config['NUM_LAYERS'],
        output_size=config['OUTPUT_SIZE']
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    
    print(f"      Architecture: LSTM({config['HIDDEN_SIZE']} units, {config['NUM_LAYERS']} layers)")
    
    # Training loop
    print(f"\n[5/5] Training for {config['NUM_EPOCHS']} epochs...")
    print("-" * 40)
    
    for epoch in range(config['NUM_EPOCHS']):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"      Epoch [{epoch+1:3d}/{config['NUM_EPOCHS']}] - Loss: {avg_loss:.6f}")
    
    print("-" * 40)
    
    # Save model and scaler
    print(f"\nSaving model to {model_path}...")
    torch.save(model.state_dict(), model_path)
    
    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)
    
    print("\n✓ Training complete!")
    
    return model, scaler

# ============================================================================
# Prediction Function
# ============================================================================
def predict(model, scaler, history_values, forecast_hours=24):
    """
    Generate predictions for future hours.
    
    Args:
        model: Trained LSTM model
        scaler: Fitted MinMaxScaler (or create new one from history)
        history_values: numpy array of historical values (at least LOOK_BACK points)
        forecast_hours: Number of hours to forecast
    
    Returns:
        Array of predicted values (inverse-transformed to original scale)
    """
    look_back = CONFIG['LOOK_BACK']
    
    if len(history_values) < look_back:
        raise ValueError(f"Need at least {look_back} history points, got {len(history_values)}")
    
    model.eval()
    
    # If no scaler provided, create one from history
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(history_values.reshape(-1, 1))
    
    # Prepare input (last LOOK_BACK points)
    input_data = history_values[-look_back:].reshape(-1, 1)
    scaled_input = scaler.transform(input_data)
    
    # Convert to tensor
    X_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 60, 1]
    
    # Iterative prediction
    forecasts = []
    current_input = X_tensor
    
    for _ in range(forecast_hours):
        with torch.no_grad():
            pred = model(current_input)  # Shape: [1, 1]
        
        forecasts.append(pred.item())
        
        # Update input: remove oldest, add new prediction
        new_step = pred.unsqueeze(1)  # Shape: [1, 1, 1]
        current_input = torch.cat((current_input[:, 1:, :], new_step), dim=1)
    
    # Inverse transform to original scale
    forecasts = np.array(forecasts).reshape(-1, 1)
    predictions = scaler.inverse_transform(forecasts).flatten()
    
    # Ensure non-negative values
    predictions = np.maximum(predictions, 0)
    
    return predictions

# ============================================================================
# Complete Forecasting Pipeline
# ============================================================================
def forecast_metric(history_values, model_path='lstm_model.pth', forecast_hours=24):
    """
    Complete forecasting pipeline for any metric.
    
    Args:
        history_values: Array of historical values (hourly data, at least 60 points)
        model_path: Path to trained model file
        forecast_hours: Number of hours to forecast
    
    Returns:
        Dictionary with history and forecast data
    """
    # Load model
    model = LSTMModel(
        input_size=1,
        hidden_size=CONFIG['HIDDEN_SIZE'],
        num_layers=CONFIG['NUM_LAYERS'],
        output_size=CONFIG['OUTPUT_SIZE']
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Create scaler from history data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(history_values.reshape(-1, 1))
    
    # Generate predictions
    predictions = predict(model, scaler, history_values, forecast_hours)
    
    return predictions

# ============================================================================
# Demo Function
# ============================================================================
def run_demo():
    """Run a demonstration of the forecasting model."""
    
    print("=" * 60)
    print("LSTM Time Series Forecasting - Demo")
    print("=" * 60)
    
    # Generate synthetic data (simulating hourly response times)
    print("\n[1/4] Generating synthetic data...")
    np.random.seed(42)
    
    hours = 200
    base_value = 100  # Base response time in ms
    
    # Create realistic pattern: daily seasonality + trend + noise
    t = np.arange(hours)
    daily_pattern = 30 * np.sin(2 * np.pi * t / 24)  # Daily cycle
    trend = 0.1 * t  # Slight upward trend
    noise = np.random.normal(0, 10, hours)  # Random noise
    
    data = base_value + daily_pattern + trend + noise
    data = np.maximum(data, 10)  # Ensure positive values
    
    print(f"      Generated {hours} hourly data points")
    print(f"      Range: {data.min():.1f} - {data.max():.1f}")
    
    # Prepare data for training
    print("\n[2/4] Preparing training data...")
    values = data.astype('float32')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values.reshape(-1, 1))
    
    X, y = create_sequences(scaled_data, CONFIG['LOOK_BACK'])
    print(f"      Created {len(X)} training sequences")
    
    # Train a smaller model for demo
    print("\n[3/4] Training model (reduced epochs for demo)...")
    
    dataset = TimeSeriesDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(30):  # Reduced epochs for demo
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/30 - Loss: {loss.item():.6f}")
    
    # Make predictions
    print("\n[4/4] Generating forecast...")
    predictions = predict(model, scaler, values, forecast_hours=24)
    
    print("\n" + "=" * 60)
    print("FORECAST RESULTS (Next 24 Hours)")
    print("=" * 60)
    
    now = datetime.now()
    for i, pred in enumerate(predictions):
        future_time = now + timedelta(hours=i+1)
        print(f"      {future_time.strftime('%H:%M')} - {pred:.1f} ms")
    
    print("\n" + "-" * 60)
    print(f"Average Forecast: {predictions.mean():.1f} ms")
    print(f"Min Forecast:     {predictions.min():.1f} ms")
    print(f"Max Forecast:     {predictions.max():.1f} ms")
    print("-" * 60)
    
    # Trend analysis
    if predictions[-1] > predictions[0]:
        trend = "↗ Rising"
    elif predictions[-1] < predictions[0]:
        trend = "↘ Falling"
    else:
        trend = "→ Stable"
    
    print(f"Trend: {trend}")
    print("\n✓ Demo complete!")

# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='LSTM Time Series Forecasting Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python forecast_model.py demo                          # Run demo with synthetic data
  python forecast_model.py train --data data.json        # Train on your data
  python forecast_model.py predict --hours 48            # Predict next 48 hours
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo with synthetic data')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model on data')
    train_parser.add_argument('--data', required=True, help='Path to JSON data file')
    train_parser.add_argument('--model', default='lstm_model.pth', help='Output model file')
    train_parser.add_argument('--scaler', default='scaler.pkl', help='Output scaler file')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    
    # Predict command  
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', default='lstm_model.pth', help='Model file')
    predict_parser.add_argument('--data', help='JSON data file for history')
    predict_parser.add_argument('--hours', type=int, default=24, help='Hours to forecast')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        run_demo()
    
    elif args.command == 'train':
        config = CONFIG.copy()
        config['NUM_EPOCHS'] = args.epochs
        train_model(args.data, args.model, args.scaler, config)
    
    elif args.command == 'predict':
        if not args.data:
            print("Error: --data is required for prediction")
            return
        
        df = load_data_from_json(args.data)
        values = df['value'].values.astype('float32')
        
        predictions = forecast_metric(values, args.model, args.hours)
        
        print(f"\nForecast for next {args.hours} hours:")
        for i, pred in enumerate(predictions):
            print(f"  Hour {i+1}: {pred:.2f}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
