# #!/bin/bash

# 1. Load HyperK dataset
echo "Step 1/5: Loading HyperK dataset..."
LOAD_RES=$(curl -s -X POST http://localhost:5001/load_hyperk)
DATA_PATH=$(echo $LOAD_RES | jq -r '.path')
echo "✅ Data loaded at: $DATA_PATH"

# 2. Convert CSV to NPZ format
echo "Step 2/5: Converting to NPZ format..."
docker-compose exec augmenter python -c "
import numpy as np
import pandas as pd
try:
    df = pd.read_csv('/app/data/hyperk_sample.csv')
    # Ensure labels are integers (0-4) for classification
    labels = (df['target'].values * 4).astype(int)  
    np.savez('/app/data/hyperk_sample.npz',
             images=df.drop('target',axis=1).values.reshape(-1, 20, 1),
             labels=labels)
    print('✅ Conversion successful. Unique labels:', np.unique(labels))
except Exception as e:
    print('❌ Conversion failed:', str(e))
    exit(1)
" || exit 1

# 3. Apply PuzzleMix augmentation
echo "Step 3/5: Applying PuzzleMix augmentation..."
AUG_RES=$(curl -s -X POST http://localhost:5002/augment \
  -H "Content-Type: application/json" \
  -d '{
    "type": "puzzlemix",
    "params": {"beta": 1.0},
    "input_path": "/app/data/hyperk_sample.npz",
    "output_path": "/app/data/augmented.npz"
  }')

if [ $(echo $AUG_RES | jq -r '.status') != "success" ]; then
    echo "❌ Augmentation failed:"
    echo $AUG_RES | jq
    exit 1
fi

AUG_PATH=$(echo $AUG_RES | jq -r '.output_path')
echo "✅ Augmented data saved at: $AUG_PATH"
echo "   Mix ratio: $(echo $AUG_RES | jq -r '.mix_ratio')"

# 4. Run training
echo "Step 4/5: Running training..."
TRAIN_RES=$(curl -s -X POST http://localhost:5003/quick_test \
  -H "Content-Type: application/json" \
  -d '{"data_path": "'$AUG_PATH'"}')

if [ $(echo $TRAIN_RES | jq -r '.status') != "success" ]; then
    echo "❌ Training failed:"
    echo $TRAIN_RES | jq
    exit 1
fi

echo "✅ Training completed:"
echo $TRAIN_RES | jq

# 5. Output results
echo "Step 5/5: Pipeline completed!"
echo ""
echo "📊 MLflow UI: http://localhost:5000"
echo "💾 Artifacts directory:"
docker-compose exec mlflow ls -lh /mlartifacts

# 6. Display explanation (if training succeeded)
if [ $(echo $TRAIN_RES | jq -r '.status') == "success" ]; then
    echo ""
    echo "🔍 Model Explanation:"
    echo "Blue bars: Features that increase prediction score"
    echo "Red bars: Features that decrease prediction score"
    
    # Save the plot as an image
    echo $TRAIN_RES | jq -r '.shap_plot' | cut -d',' -f2 | base64 -d > explanation.png
    echo "📊 Explanation saved as explanation.png"
    
    # For terminal display (if imgcat is available):
    if command -v imgcat &> /dev/null; then
        imgcat explanation.png
    fi
fi