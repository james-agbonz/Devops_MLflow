from flask import Flask, jsonify
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Configuration
DATA_DIR = "/data"
CLASSES = ["normal", "dyed-lifted-polyps"]
IMAGE_SIZE = (224, 224)

def process_images():
    """Load and process all images, returning summary stats"""
    image_counts = {cls: 0 for cls in CLASSES}
    shapes = []
    
    for class_name in CLASSES:
        for split in ["train", "test"]:
            dir_path = os.path.join(DATA_DIR, split, class_name)
            if not os.path.exists(dir_path):
                continue
                
            for filename in os.listdir(dir_path):
                img_path = os.path.join(dir_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, IMAGE_SIZE)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    image_counts[class_name] += 1
                    shapes.append(img.shape)
    
    return {
        "total_images": sum(image_counts.values()),
        "class_counts": image_counts,
        "unique_shapes": list(set(shapes))  # Get unique shapes found
    }

@app.route('/load', methods=['POST'])
def load_data():
    try:
        stats = process_images()
        
        return jsonify({
            "status": "success",
            "data_stats": stats,
            "message": "All images processed successfully"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/')
def home():
    return jsonify({"status": "ready"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
