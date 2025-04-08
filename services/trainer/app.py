from flask import Flask, request, jsonify
import mlflow
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import os
import json  

app = Flask(__name__)

@app.route('/health')
def health():
    return {'status': 'healthy'}, 200

def create_shap_plot(model, X):
    """Generate SHAP explanation plot"""
    explainer = shap.Explainer(model.predict_proba, X)
    shap_values = explainer(X[:5])  # Explain first 5 samples
    
    plt.figure()
    shap.plots.beeswarm(shap_values[:,:,1], show=False)
    plt.tight_layout()
    
    # Save to buffer and file
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save to file for MLflow
    os.makedirs('/app/data/shap', exist_ok=True)
    plot_path = "/app/data/shap/shap_plot.png"
    with open(plot_path, 'wb') as f:
        f.write(buf.getvalue())
    
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8'), plot_path

@app.route('/quick_test', methods=['POST'])
def quick_test():
    try:
        data = np.load(request.json['data_path'])
        X = data['images'].reshape(len(data['images']), -1)
        y = (data['labels'] * 4).astype(int)
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'model_type': 'LogisticRegression',
                'max_iter': 1000,
                'features': X.shape[1],
                'samples': X.shape[0],
                'classes': len(np.unique(y))
            })
            
            # Train model
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log metrics
            acc = model.score(X, y)
            mlflow.log_metrics({
                "accuracy": acc,
                "training_samples": len(X)
            })
            
            # Generate and log explanation
            shap_plot_base64, shap_plot_path = create_shap_plot(model, X)
            mlflow.log_artifact(shap_plot_path, "explanation")
            
            # Log sample data
            sample_path = "/app/data/train_sample.csv"
            pd.DataFrame(X[:100]).to_csv(sample_path)
            mlflow.log_artifact(sample_path)
            
            return jsonify({
                "status": "success",
                "accuracy": float(acc),
                "shap_plot": f"data:image/png;base64,{shap_plot_base64}",
                "message": "Blue=positive impact, Red=negative impact",
                "model_uri": mlflow.active_run().info.artifact_uri + "/model"
            })
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)