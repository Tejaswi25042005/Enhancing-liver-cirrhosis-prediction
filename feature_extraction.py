import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def extract_features(dataset_path, img_size=(64, 64), feature_path='features'):
    """
    Extract features from liver cancer dataset images
    
    Parameters:
    - dataset_path: Directory containing image folders
    - img_size: Resize dimension for images
    - feature_path: Directory to save extracted features
    """
    # Create features directory if not exists
    os.makedirs(feature_path, exist_ok=True)
    
    # Lists to store features and labels
    X = []
    Y = []
    
    # Label encoder to convert class names to numeric
    label_encoder = LabelEncoder()
    
    # Iterate through class subdirectories
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    classes.sort()
    labels = label_encoder.fit_transform(classes)
    
    # Process images
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        
        # Iterate through images in class directory
        for img_file in tqdm(os.listdir(class_path), desc=f'Processing {class_name}'):
            img_path = os.path.join(class_path, img_file)
            
            try:
                # Read image
                img = cv2.imread(img_path)
                
                # Resize image
                img = cv2.resize(img, img_size)
                
                # Normalize pixel values
                img = img.astype('float32') / 255.0
                
                # Append image and label
                X.append(img)
                Y.append(labels[idx])
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    
    # Save features
    np.save(os.path.join(feature_path, 'X.txt.npy'), X)
    np.save(os.path.join(feature_path, 'Y.txt.npy'), Y)
    
    print(f"Total images processed: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Labels shape: {Y.shape}")
    print(f"Unique labels: {np.unique(Y)}")
    print(f"Label mapping: {dict(zip(labels, classes))}")

# Example usage
if __name__ == "__main__":
    dataset_path = dataset_path = "H:/liver cirrhosis/New folder/live_cancer_dataset"
    extract_features(dataset_path)