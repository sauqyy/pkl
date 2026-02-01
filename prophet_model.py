
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)

def train_predict(history_values, history_timestamps, forecast_days=7, metric_name='Response'):
    """
    Train a Prophet model and generate forecasts using logic from PKL.ipynb.
    - Resamples to Daily.
    - Uses specific Prophet params.
    """
    try:
        # Prepare DataFrame for Prophet
        df = pd.DataFrame({
            'ds': pd.to_datetime(history_timestamps, unit='ms'),
            'y': history_values
        })
        
        # Sort by date
        df = df.sort_values('ds').set_index('ds')
        
        # Resample to Hourly (H) to match user request for "7 am data, 8 am data"
        if metric_name == 'Response':
            df = df['y'].resample('H').mean()
        else:
            df = df['y'].resample('H').sum()
            
        df = df.fillna(0).reset_index()
        
        # Note: Notebook uses:
        # model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, interval_width=0.9)
        # But we enable daily_seasonality for hourly data.
        
        # Initialize model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True, # Enabled for hourly
            interval_width=0.9
        )
        model.fit(df)

        # Future Forecast (Hourly)
        future = model.make_future_dataframe(periods=forecast_days * 24, freq='H') 
        forecast = model.predict(future)
        
        # Extract comparison (actual vs predicted where overlapping)
        comparison = []
        forecast_result = []
        
        merged = pd.merge(df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner')
        y_true = merged['y'].values
        y_pred = merged['yhat'].values
        y_pred = np.maximum(y_pred, 0)
        
        mae = mean_absolute_error(y_true, y_pred)
        # Avoid division by zero for MAPE
        mask = y_true != 0
        mape = 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        for index, row in merged.iterrows():
            comparison.append({
                'timestamp': int(row['ds'].timestamp() * 1000),
                'actual': float(row['y']),
                'predicted': float(max(0, row['yhat']))
            })
        
        # Limit history to last 7 days (hourly points) to focus on forecast
        comparison = comparison[-(forecast_days * 24):]

        # Future part only
        last_history_date = df['ds'].max()
        future_only = forecast[forecast['ds'] > last_history_date].head(forecast_days * 24)
        
        for index, row in future_only.iterrows():
            forecast_result.append({
                'timestamp': int(row['ds'].timestamp() * 1000),
                'value': float(max(0, row['yhat']))
            })
            
        return {
            'comparison': comparison,
            'forecast': forecast_result,
            'accuracy': {
                'mae': float(mae),
                'mape': float(mape)
            },
            'model': 'Prophet (Hourly)'
        }

    except Exception as e:
        logger.error(f"Prophet forecasting failed: {e}")
        # Return empty structure or re-raise
        raise e
