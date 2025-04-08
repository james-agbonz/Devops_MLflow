import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

SERVICES = {
    "data_loader": "http://localhost:5001",
    "augmenter": "http://localhost:5002",
    "trainer": "http://localhost:5003", 
    "evaluator": "http://localhost:5004"
}

def wait_for_service(url, timeout=30):
    """Wait for a service to become available"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(f"{url}/health").status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    return False

def test_data_loader():
    print("\n=== Testing Data Loader ===")
    res = requests.post(
        f"{SERVICES['data_loader']}/load",
        json={"file_path": os.getenv("DATA_INPUT_PATH")}
    )
    print(f"Status: {res.status_code}")
    print(f"Response: {res.json()}")
    return res.json()['path']

def test_augmenter(input_path):
    print("\n=== Testing Augmenter ===")
    res = requests.post(
        f"{SERVICES['augmenter']}/augment",
        json={
            "type": "basic",
            "params": {
                "rotate": int(os.getenv("AUG_ROTATION")),
                "flip": os.getenv("AUG_FLIP") == "True",
                "brightness": float(os.getenv("AUG_BRIGHTNESS_ADJUST"))
            },
            "input_path": input_path,
            "output_path": os.getenv("DATA_OUTPUT_PATH")
        }
    )
    print(f"Status: {res.status_code}")
    print(f"Response: {res.json()}")
    return res.json()['output_path']

def test_trainer(data_path):
    print("\n=== Testing Trainer ===")
    res = requests.post(
        f"{SERVICES['trainer']}/train",
        json={"data_path": data_path}
    )
    print(f"Status: {res.status_code}")
    print(f"Response: {res.json()}")
    return res.json()['model_uri']

def test_evaluator(model_uri, test_data_path):
    print("\n=== Testing Evaluator ===")
    res = requests.post(
        f"{SERVICES['evaluator']}/evaluate",
        json={
            "data_path": test_data_path,
            "model_uri": model_uri
        }
    )
    print(f"Status: {res.status_code}")
    print("Metrics:")
    for metric, value in res.json()['metrics'].items():
        print(f"- {metric}: {value:.4f}")
    return res.json()

if __name__ == "__main__":
    print("Waiting for services to start...")
    for name, url in SERVICES.items():
        if wait_for_service(url):
            print(f"{name} is ready")
        else:
            print(f"⚠️ {name} failed to start")
            exit(1)

    try:
        # Test pipeline components
        processed_path = test_data_loader()
        augmented_path = test_augmenter(processed_path)
        model_uri = test_trainer(augmented_path)
        test_evaluator(model_uri, os.getenv("TEST_DATA_PATH"))
        
        print("\n✅ All tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        exit(1)