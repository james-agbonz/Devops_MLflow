from flask import Flask, request, jsonify
import pandas as pd
import os
import numpy as np
import json 

app = Flask(__name__)

@app.route('/load_hyperk', methods=['POST'])
def load_hyperk():
    try:
        # Create simple dummy data if real data isn't available
        X = np.random.rand(100, 20)  # 100 samples, 20 features
        y = np.random.randint(0, 5, 100)  # 5 classes
        
        df = pd.DataFrame(X)
        df['target'] = y
        
        os.makedirs('/app/data', exist_ok=True)
        output_path = "/app/data/hyperk_sample.csv"
        df.to_csv(output_path, index=False)
        
        return jsonify({
            "status": "success",
            "path": output_path,
            "samples": len(df)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# ***DISCRETE VALUES

# @app.route('/load_hyperk', methods=['POST'])
# def load_hyperk():
#     try:
#         # Create synthetic data with INTEGER labels
#         from sklearn.datasets import make_classification
#         X, y = make_classification(n_samples=100, n_features=20, n_classes=5, n_informative=5)
        
#         df = pd.DataFrame(X)
#         df['target'] = y  # Integer labels (0-4)
        
#         output_path = "/app/data/hyperk_sample.csv"
#         df.to_csv(output_path, index=False)
        
#         return jsonify({
#             "status": "success",
#             "path": output_path,
#             "samples": len(df),
#             "classes": int(y.max()) + 1  # Number of classes
#         })
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500