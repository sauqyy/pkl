
import requests
import time
import sys

# Replace with your bot token
BOT_TOKEN = "8290043825:AAFeZVa2F8kBXJduCktSJOJ162SRF9QDhFM"
URL = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"

def get_chat_id():
    try:
        response = requests.get(URL)
        data = response.json()
        
        if not data.get("ok"):
            print("Error: API returned 'ok': false")
            print(data)
            return

        updates = data.get("result", [])
        if not updates:
            print("No updates found. Please send a message (e.g., /start) to your bot first.")
            return

        # Get the latest update
        latest_update = updates[-1]
        
        chat_id = None
        username = "Unknown"
        
        if "message" in latest_update:
            chat_id = latest_update["message"]["chat"]["id"]
            username = latest_update["message"]["chat"].get("username", "Unknown")
        elif "my_chat_member" in latest_update:
            chat_id = latest_update["my_chat_member"]["chat"]["id"]
            username = latest_update["my_chat_member"]["chat"].get("username", "Unknown")
            
        print(f"\nFound Chat ID: {chat_id}")
        print(f"Username: {username}")
        print("\nPlease copy this Chat ID and update your app.py configuration.")
        return chat_id

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print("Checking for Telegram updates...")
    get_chat_id()
