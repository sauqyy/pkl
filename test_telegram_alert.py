
import unittest
import requests
from unittest.mock import patch, MagicMock
from app import send_telegram_alert, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

class TestTelegramAlert(unittest.TestCase):

    @patch('app.TELEGRAM_CHAT_ID', '5958836175')
    @patch('requests.post')
    def test_send_telegram_alert_success(self, mock_post):
        # Configure the mock to return a successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Call the function
        message = "Test Error Message"
        send_telegram_alert(message)

        # Verify requests.post was called with correct arguments
        expected_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        expected_data = {
            "chat_id": '5958836175',
            "text": f"🚨 *AppDynamics Alert*\n\n{message}",
            "parse_mode": "Markdown"
        }
        mock_post.assert_called_once_with(expected_url, json=expected_data)

    @patch('app.TELEGRAM_CHAT_ID', '123456789')
    @patch('requests.post')
    def test_send_telegram_alert_failure(self, mock_post):
        # Configure the mock to raise an exception
        mock_post.side_effect = requests.exceptions.RequestException("Network Error")

        # Call the function (should catch exception and print error, not crash)
        try:
            send_telegram_alert("Test Error")
        except Exception:
            self.fail("send_telegram_alert raised Exception unexpectedly!")

if __name__ == '__main__':
    unittest.main()
