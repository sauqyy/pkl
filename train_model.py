import json
import numpy as np
import pandas as pd
import joblib 
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# Config
# Config
DATA_FILE = 'training_data.json'
MODEL_DIR = '.'  # Directory to save models

def get_model_paths(metric_name='response'):
    """Get paths for model and scaler based on metric name."""
    clean_name = metric_name.lower().strip()
    return f'lstm_model_{clean_name}.pth', f'scaler_{clean_name}.pkl'

MODEL_FILE = 'lstm_model.pth' # Default for backward compatibility
SCALER_FILE = 'scaler.pkl'
LOOK_BACK = 60 
HIDDEN_SIZE = 50
NUM_LAYERS = 2
OUTPUT_SIZE = 1
NUM_EPOCHS = 100 
BATCH_SIZE = 16
LEARNING_RATE = 0.001

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

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def train_metric_model(data_points, metric_name='response'):
    """
    Train a model for a specific metric.
    data_points: list of dicts with 'value' and 'startTimeInMillis'
    """
    model_path, scaler_path = get_model_paths(metric_name)
    lock_file = f'training_{metric_name}.lock'
    
    # Indicate training start
    with open(lock_file, 'w') as lock:
        lock.write('training')

    try:
        print(f"Training model for {metric_name}...")
        
        # DataFrame
        df = pd.DataFrame(data_points)
        if 'timestamp' in df.columns:
             df = df.rename(columns={'timestamp': 'startTimeInMillis'})
             
        df = df.sort_values('startTimeInMillis')
        
        # Use 'value'
        data = df[['value']].values.astype('float32')
        
        if len(data) < LOOK_BACK + 10:
            print(f"Not enough data to train {metric_name}")
            return False
        
        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Create Sequences
        X, y = create_dataset(scaled_data, LOOK_BACK)
        if len(X) == 0:
            return False
            
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Dataset/Loader
        dataset = TimeSeriesDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Model
        model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        print(f"Training PyTorch LSTM for {metric_name} with {len(X)} samples...")
        
        # Train
        for epoch in range(NUM_EPOCHS):
            model.train()
            loss_val = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
            
            if (epoch+1) % 10 == 0:
                print(f'[{metric_name}] Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss_val:.4f}')
                
        # Save
        print(f"Saving {metric_name} model to {model_path}...")
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)
        print(f"Done training {metric_name}.")
        return True

    except Exception as e:
        print(f"Error training {metric_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Remove lock file
        if os.path.exists(lock_file):
            os.remove(lock_file)

def main():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            raw_data = json.load(f)
        train_metric_model(raw_data, 'response')
    else:
        print(f"{DATA_FILE} not found.")

if __name__ == "__main__":
    main()
