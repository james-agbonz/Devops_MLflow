#!/usr/bin/env python3
# Save this as convert_csv_to_npz.py

import numpy as np
import pandas as pd
import os
import sys

try:
    # Check if the input file exists
    input_path = '/app/data/hyperk_sample.csv'
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file {input_path} does not exist")
        sys.exit(1)
        
    print(f"Found input file, size: {os.path.getsize(input_path)} bytes")
    
    # Load the CSV file with error handling
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded CSV with shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error reading CSV file: {str(e)}")
        sys.exit(1)
    
    # Verify required columns
    if 'target' not in df.columns:
        print(f"❌ Error: 'target' column not found in CSV. Available columns: {df.columns.tolist()}")
        sys.exit(1)
        
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname('/app/data/hyperk_sample.npz')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Process data safely
    try:
        # Extract features and labels
        features = df.drop('target', axis=1).values
        labels = df['target'].values
        
        # Print information about the data
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels before conversion: {labels[:5]} (showing first 5)")
        
        # Convert labels to integers
        labels = (labels * 4).astype(int)
        print(f"Labels after conversion: {labels[:5]} (showing first 5)")
        print(f"Unique labels: {np.unique(labels)}")
        
        # Reshape features for deep learning
        # Check if reshape is possible
        if features.size != features.shape[0] * 20:
            print(f"❌ Error: Cannot reshape features array of shape {features.shape} to include dimension 20")
            print(f"Expected {features.shape[0] * 20} elements but got {features.size}")
            sys.exit(1)
            
        reshaped_features = features.reshape(-1, 20, 1)
        print(f"Reshaped features to: {reshaped_features.shape}")
        
        # Save to NPZ file
        output_path = '/app/data/hyperk_sample.npz'
        np.savez(output_path, images=reshaped_features, labels=labels)
        print(f"✅ Conversion successful. Saved to {output_path}")
        print(f"Output file size: {os.path.getsize(output_path)} bytes")
        
    except Exception as e:
        print(f"❌ Error during data processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Unexpected error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)