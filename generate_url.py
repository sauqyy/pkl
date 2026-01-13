import time
from datetime import datetime
import urllib.parse

def get_appdynamics_url(base_url, application, metric_path, start_time_str, end_time_str):
    """
    Generates a full AppDynamics REST API URL for a specific time range.
    Time format: YYYY-MM-DD HH:MM:SS
    """
    # Convert string time to Unix timestamp in milliseconds
    dt_format = "%Y-%m-%d %H:%M:%S"
    start_dt = datetime.strptime(start_time_str, dt_format)
    end_dt = datetime.strptime(end_time_str, dt_format)
    
    # Epoch in milliseconds
    start_epoch = int(start_dt.timestamp() * 1000)
    end_epoch = int(end_dt.timestamp() * 1000)
    
    # URL Encoding the metric path
    encoded_metric = urllib.parse.quote(metric_path)
    
    # Construct URL
    # Template: /controller/rest/applications/{application}/metric-data
    # rollup=false forces AppDynamics to give the finest granularity available (usually 1 minute)
    full_url = (
        f"{base_url}/controller/rest/applications/{application}/metric-data?"
        f"metric-path={encoded_metric}&"
        f"time-range-type=BETWEEN_TIMES&"
        f"start-time={start_epoch}&"
        f"end-time={end_epoch}&"
        f"output=JSON&"
        f"rollup=false" 
    )
    
    return full_url

# --- Konfigurasi Anda ---
BASE_URL = "https://ptfwdindonesia-prod.saas.appdynamics.com"
APP_NAME = "SmartNano"
METRIC_PATH = "Overall Application Performance|Average Response Time (ms)"

# CATATAN PENTING:
# API Metrics AppDynamics biasanya hanya menyimpan data per 1 MENIT (bukan per detik).
# Dengan `rollup=false`, Anda akan mendapatkan data poin per menit.
# Jika Anda meminta rentang waktu yang pendek (misal 1 jam), Anda akan dapat ~60 poin data.

START_TIME = "2026-01-12 10:00:00"  # Format: YYYY-MM-DD HH:MM:SS
END_TIME   = "2026-01-12 11:00:00"

print(f"--- Generating URL for {START_TIME} to {END_TIME} ---")
generated_url = get_appdynamics_url(BASE_URL, APP_NAME, METRIC_PATH, START_TIME, END_TIME)
print("\nSalin URL ini ke browser Anda (jika sudah login) atau gunakan di script dengan Auth yang benar:")
print(generated_url)
