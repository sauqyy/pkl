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
DATA_FILE = 'training_data.json'
MODEL_FILE = 'lstm_model.pth'
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

def main():
    # Indicate training start
    with open('training.lock', 'w') as lock:
        lock.write('training')

    try:
        print("Loading data...")
        if not os.path.exists(DATA_FILE):
            print(f"Error: {DATA_FILE} not found.")
            return

        with open(DATA_FILE, 'r') as f:
            raw_data = json.load(f)

        # DataFrame
        df = pd.DataFrame(raw_data)
        df = df.sort_values('startTimeInMillis')
        
        # Use 'value'
        data = df[['value']].values.astype('float32')
        
        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Create Sequences
        X, y = create_dataset(scaled_data, LOOK_BACK)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Dataset/Loader
        dataset = TimeSeriesDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Model
        model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        print(f"Training PyTorch LSTM with {len(X)} samples over {NUM_EPOCHS} epochs...")
        
        # Train
        for epoch in range(NUM_EPOCHS):
            model.train()
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')
                
        # Save
        print("Saving model and scaler...")
        torch.save(model.state_dict(), MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        print("Done.")

    finally:
        # Remove lock file
        if os.path.exists('training.lock'):
            os.remove('training.lock')

if __name__ == "__main__":
    main()
