#!/usr/bin/env python3
# Save this as test_mlflow.py

import mlflow
import sys
import os

print("Testing MLflow connection...")

# Set tracking URI
tracking_uri = "http://localhost:5000"
print(f"Setting tracking URI to: {tracking_uri}")
mlflow.set_tracking_uri(tracking_uri)

try:
    # Try to connect to the tracking server
    client = mlflow.tracking.MlflowClient()
    print("Successfully created MLflow client")
    
    # List experiments to test connection
    experiments = client.list_experiments()
    print(f"Found {len(experiments)} experiments")
    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")
    
    # Create a test run
    print("\nCreating a test run...")
    with mlflow.start_run(run_name="Connection Test") as run:
        run_id = run.info.run_id
        print(f"Created run with ID: {run_id}")
        
        # Log some parameters and metrics
        mlflow.log_param("test_param", "test_value")
        mlflow.log_metric("test_metric", 1.0)
        
        # Create and log a test artifact
        artifact_path = "/tmp/test_artifact.txt"
        with open(artifact_path, "w") as f:
            f.write("This is a test artifact")
        
        mlflow.log_artifact(artifact_path)
        print(f"Logged test artifact: {artifact_path}")
    
    print("\nMLflow connection test completed successfully!")
    print(f"You can view this run at: {tracking_uri}/#/experiments/0/runs/{run_id}")
    
except Exception as e:
    print(f"❌ MLflow connection test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)