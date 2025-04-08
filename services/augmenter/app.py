from flask import Flask, request, jsonify
import numpy as np
from techniques import apply_augmentation
import mlflow
import json
import os

app = Flask(__name__)

@app.route('/augment', methods=['POST'])
def augment():
    try:
        config = request.get_json()
        print("Received config:", config)  # Debug log
        
        # Validate required fields
        required = ['type', 'input_path', 'output_path']
        if not all(field in config for field in required):
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {required}"
            }), 400

        # Initialize defaults
        config.setdefault('params', {'beta': 1.0})
        
        # Verify input file exists
        if not os.path.exists(config['input_path']):
            return jsonify({
                "status": "error",
                "message": f"Input file not found: {config['input_path']}"
            }), 400

        with mlflow.start_run():
            result = apply_augmentation(config)
            
            if result['status'] == 'success':
                mlflow.log_params({
                    'augmentation_type': config['type'],
                    'beta': config['params']['beta']
                })
                mlflow.log_metric('mix_ratio', result['mix_ratio'])
                mlflow.log_artifact(config['input_path'], 'input')
                mlflow.log_artifact(result['output_path'], 'output')
            
            return jsonify(result)
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)