
from app import send_telegram_alert

if __name__ == "__main__":
    print("Sending test alert to Telegram...")
    send_telegram_alert("This is a test alert from your AppDynamics Dashboard! 🚀\n\nIf you see this, the integration is working successfully.")
    print("Check your Telegram!")
