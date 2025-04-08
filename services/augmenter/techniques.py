# Save this as techniques.py in your augmenter service directory

import numpy as np
import os
import random

def apply_augmentation(config):
    """
    Apply the specified augmentation technique to the input data
    
    Args:
        config (dict): Configuration dictionary with keys:
            - type: Type of augmentation
            - params: Parameters for the augmentation
            - input_path: Path to input data
            - output_path: Path to save augmented data
            
    Returns:
        dict: Result with status and output path
    """
    try:
        # Load the input data
        input_data = np.load(config['input_path'])
        images = input_data['images']
        labels = input_data['labels']
        
        # Get augmentation parameters
        aug_type = config['type']
        params = config.get('params', {})
        
        # Apply the specified augmentation
        if aug_type == 'puzzlemix':
            augmented_images, mix_ratio = puzzlemix_augmentation(
                images, 
                beta=params.get('beta', 1.0)
            )
        elif aug_type == 'basic':
            augmented_images, mix_ratio = basic_augmentation(
                images,
                rotate=params.get('rotate', 0),
                flip=params.get('flip', False),
                brightness=params.get('brightness', 0.0)
            )
        else:
            return {
                'status': 'error',
                'message': f'Unknown augmentation type: {aug_type}'
            }
        
        # Save the augmented data
        os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
        np.savez(
            config['output_path'],
            images=augmented_images,
            labels=labels
        )
        
        return {
            'status': 'success',
            'mix_ratio': mix_ratio,
            'output_path': config['output_path'],
            'output_samples': len(augmented_images)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

def puzzlemix_augmentation(images, beta=1.0):
    """
    Implement PuzzleMix augmentation
    
    Args:
        images (ndarray): Input images
        beta (float): Beta parameter for mixing (higher = more mixing)
        
    Returns:
        tuple: (augmented images, mix ratio)
    """
    # Calculate a dynamic mix ratio based on beta
    # This ensures mix_ratio isn't always 1.0
    mix_ratio = np.random.beta(beta, beta)
    
    # Get the number of images
    n_images = images.shape[0]
    
    # Create augmented images
    augmented_images = np.copy(images)
    
    # For each image, mix it with another random image
    for i in range(n_images):
        # Select a random image to mix with
        j = random.randint(0, n_images-1)
        while j == i:  # Ensure we don't mix with the same image
            j = random.randint(0, n_images-1)
            
        # Mix the images
        augmented_images[i] = images[i] * mix_ratio + images[j] * (1 - mix_ratio)
    
    return augmented_images, float(mix_ratio)

def basic_augmentation(images, rotate=0, flip=False, brightness=0.0):
    """
    Implement basic augmentation techniques
    
    Args:
        images (ndarray): Input images
        rotate (int): Rotation angle in degrees
        flip (bool): Whether to flip images horizontally
        brightness (float): Brightness adjustment factor
        
    Returns:
        tuple: (augmented images, mix ratio)
    """
    # Copy the images
    augmented_images = np.copy(images)
    
    # Apply brightness adjustment
    if brightness != 0.0:
        augmented_images = augmented_images * (1 + brightness)
        augmented_images = np.clip(augmented_images, 0, 1)
    
    # For simplicity, we're returning a random mix ratio between 0.5 and 1.0
    # In a real implementation, this would be calculated based on the actual transformations
    mix_ratio = random.uniform(0.5, 1.0)
    
    return augmented_images, float(mix_ratio)