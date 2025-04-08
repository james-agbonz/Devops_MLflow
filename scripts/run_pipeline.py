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
    """Wait for service to become available"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(f"{url}/health").status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    return False

def run_pipeline():
    print("🚀 Starting ML Pipeline")
    
    # 1. Load Data
    print("\n📂 [1/4] Loading data...")
    load_res = requests.post(
        f"{SERVICES['data_loader']}/load",
        json={"file_path": os.getenv("DATA_INPUT_PATH")}
    )
    load_res.raise_for_status()
    processed_path = load_res.json()['path']
    print(f"Data loaded and processed to: {processed_path}")

    # 2. Augment Data
    print("\n✨ [2/4] Augmenting data...")
    augment_res = requests.post(
        f"{SERVICES['augmenter']}/augment",
        json={
            "type": "basic",
            "params": {
                "rotate": int(os.getenv("AUG_ROTATION")),
                "flip": os.getenv("AUG_FLIP") == "True",
                "brightness": float(os.getenv("AUG_BRIGHTNESS_ADJUST"))
            },
            "input_path": processed_path,
            "output_path": os.getenv("DATA_OUTPUT_PATH")
        }
    )
    augment_res.raise_for_status()
    augmented_path = augment_res.json()['output_path']
    print(f"Augmented data saved to: {augmented_path}")

    # 3. Train Model
    print("\n🤖 [3/4] Training model...")
    train_res = requests.post(
        f"{SERVICES['trainer']}/train",
        json={"data_path": augmented_path}
    )
    train_res.raise_for_status()
    model_uri = train_res.json()['model_uri']
    print(f"Model trained and saved to: {model_uri}")

    # 4. Evaluate Model
    print("\n📊 [4/4] Evaluating model...")
    eval_res = requests.post(
        f"{SERVICES['evaluator']}/evaluate",
        json={
            "data_path": os.getenv("TEST_DATA_PATH"),
            "model_uri": model_uri
        }
    )
    eval_res.raise_for_status()
    print("\nEvaluation Results:")
    for metric, value in eval_res.json()['metrics'].items():
        print(f"- {metric}: {value:.4f}")
    
    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    # Verify services are ready
    print("🔍 Checking services...")
    for name, url in SERVICES.items():
        if wait_for_service(url):
            print(f"✔ {name} is ready")
        else:
            print(f"✖ {name} failed to start")
            exit(1)

    try:
        run_pipeline()
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        exit(1)