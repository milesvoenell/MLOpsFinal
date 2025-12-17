import requests
import json

# Replace with your EC2 public IP
BASE_URL = "http://ec2-18-232-31-184.compute-1.amazonaws.com:8000"

# Example input
data = {
    "features": [
        {"feature1": 5.1, "feature2": 3.5, "feature3": 1.4, "feature4": 0.2}
    ]
}

endpoints = ["/predict_model1", "/predict_model2", "/predict_model3"]

for endpoint in endpoints:
    url = BASE_URL + endpoint
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"{endpoint} SUCCESS:\n{json.dumps(response.json(), indent=4)}\n")
        else:
            print(f"{endpoint} ERROR {response.status_code}:\n{response.json()}\n")
    except Exception as e:
        print(f"{endpoint} EXCEPTION:\n{str(e)}\n")
