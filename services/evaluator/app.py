from flask import Flask, request, jsonify
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json  

app = Flask(__name__)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        eval_config = request.json
        data_path = eval_config['data_path']
        model_uri = eval_config['model_uri']
        
        # Load data and model
        df = pd.read_csv(data_path)
        X = df.drop('target', axis=1)
        y_true = df['target']
        model = mlflow.sklearn.load_model(model_uri)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Generate confusion matrix visualization
        os.makedirs('/app/metrics', exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        
        # Save as JSON
        cm_json_path = "/app/metrics/confusion_matrix.json"
        with open(cm_json_path, 'w') as f:
            json.dump(cm.tolist(), f)
        
        # Save as PNG
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        cm_png_path = "/app/metrics/confusion_matrix.png"
        plt.savefig(cm_png_path, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_params({
                'eval_data_path': data_path,
                'model_uri': model_uri,
                'dataset_samples': len(df),
                'num_features': X.shape[1]
            })
            
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(cm_json_path)
            mlflow.log_artifact(cm_png_path)
            mlflow.log_dict(metrics, "metrics.json")
        
        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'confusion_matrix': cm_png_path,
            'artifacts_uri': mlflow.active_run().info.artifact_uri
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)