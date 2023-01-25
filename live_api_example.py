import requests
import json

if __name__ == "__main__":
    
    r = requests.get("https://census-salary-model.herokuapp.com/")
    print(f"Root endpoint (GET) status code: {r.status_code}")
    print(f"Root endpoint (GET) content: {r.text}")
    
    
    r = requests.get("https://census-salary-model.herokuapp.com/health")
    print(f"Health endpoint (GET) status code: {r.status_code}")
    print(f"Health endpoint (GET) content: {r.text}")

    d = {
        "inputs": [
            {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        ]
    }
    
    r = requests.post("https://census-salary-model.herokuapp.com/predict", data=json.dumps(d))
    print(f"Predict endpoint (POST) status code: {r.status_code}")
    print(f"Predict endpoint (POST) content: {r.text}")