import urllib.request
import base64
import json

url = "https://ptfwdindonesia-prod.saas.appdynamics.com/controller/rest/applications/SmartNano/metric-data?metric-path=Overall%20Application%20Performance%7CAverage%20Response%20Time%20%28ms%29&time-range-type=BETWEEN_TIMES&start-time=1768183200120&end-time=1768212000120&output=JSON"
username = "rai.archv@gmail.com@ptfwdindonesia-prod"
password = "Tester12345*"

candidates = [
    "rai.archv@gmail.com",
    "rai.archv@gmail.com@ptfwdindonesia-prod",
    "rai.archv@ptfwdindonesia-prod", 
    "rai.archv@gmail.com@ptfwdindonesia",
    "rai.archv@ptfwdindonesia"
]

for username in candidates:
    print(f"Trying User: {username} ...")
    request = urllib.request.Request(url)
    base64string = base64.b64encode(f"{username}:{password}".encode()).decode().replace('\n', '')
    request.add_header("Authorization", f"Basic {base64string}")

    try:
        with urllib.request.urlopen(request) as response:
            data = response.read().decode()
            print(f"SUCCESS with {username}!")
            # print(data) 
            break
    except Exception as e:
        print(f"Failed: {e}")
