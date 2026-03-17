Liver Tumor Classification using SVM & KNN

Overview

This project focuses on detecting liver tumors from CT scan images using machine learning techniques. The system classifies images into Normal and Tumor categories using Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) algorithms. A simple GUI is provided for real-time prediction.

Features

- Image classification (Normal vs Tumor)
- Feature extraction using OpenCV
- Machine Learning models: SVM and KNN
- GUI-based prediction system

Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-learn
- Tkinter

Dataset

- Contains CT scan images categorized into:
  - "normal"
  - "tumor"
- Includes corresponding mask images
- Note: Only a subset of the dataset is uploaded due to size limitations

Project Workflow

1. Load dataset from folders
2. Resize images to 64x64
3. Extract features using image processing
4. Train SVM and KNN models
5. Predict output using GUI

How to Run

1. Install required libraries:
   pip install numpy opencv-python scikit-learn tqdm
2. Run feature extraction:
   python feature_extraction.py
3. Run the application:
   python project.py

Output

- Classifies input image as Normal or Tumor
- Displays prediction result in GUI

Author

Tejaswi Yadlapalli
