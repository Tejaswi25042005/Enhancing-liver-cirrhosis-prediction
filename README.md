# Enhancing-Liver-Cirrhosis-Images-through-X-ray


# 🩺 Enhancing Liver Cirrhosis Detection Using SVM and KNN

## 📖 Overview
This project focuses on developing an intelligent and automated diagnostic system for the early detection of **Liver Cirrhosis** using **X-ray image processing** and **Machine Learning** techniques.  
The system utilizes two powerful algorithms — **Support Vector Machine (SVM)** and **K-Nearest Neighbors (KNN)** — to classify liver images as *normal* or *cirrhotic*, thereby assisting medical professionals in accurate diagnosis.

---

## 🎯 Objectives
- Enhance the detection of liver cirrhosis through X-ray images using ML-based image analysis.  
- Compare the performance and accuracy of **SVM** and **KNN** models.  
- Reduce manual dependency and human error in medical diagnosis.  
- Provide an automated, non-invasive, and cost-effective diagnostic tool.  

---

## 💡 Motivation
Liver cirrhosis is a major cause of mortality worldwide.  
Traditional methods like biopsy or manual image inspection are time-consuming, invasive, and error-prone.  
This project introduces a **machine learning–based approach** that automates the detection process, helping doctors identify the disease early and accurately.

---

## ⚙️ System Workflow
1. **Image Data Collection** – X-ray images of the liver are collected and labeled.  
2. **Preprocessing** – Images are enhanced, denoised, and standardized using OpenCV.  
3. **Feature Extraction** – Texture, edge, and intensity features are extracted from the images.  
4. **Model Training** – SVM and KNN algorithms are trained on the processed features.  
5. **Evaluation** – Accuracy, precision, recall, and F1-score are calculated to compare performance.  
6. **Visualization** – Graphs and confusion matrices visualize model performance.  

---

## 🧠 Algorithms Used
### 1. **Support Vector Machine (SVM)**
- A supervised learning algorithm used for classification tasks.  
- Works effectively on high-dimensional data and separates classes using hyperplanes.

### 2. **K-Nearest Neighbors (KNN)**
- A simple distance-based algorithm that classifies data based on the nearest neighbors.  
- Used here to compare with SVM for performance evaluation.

---

## 🧰 Software & Hardware Requirements

### Software:
- **Language:** Python 3.12  
- **Libraries:** scikit-learn, OpenCV, NumPy, Pandas, Matplotlib, Seaborn  
- **IDE:** Jupyter Notebook / PyCharm / VS Code  

### Hardware:
- Minimum **8 GB RAM** (16 GB recommended)  
- **Intel i5/i7** processor  
- **Full HD Display (1920×1080)**  

---

## 🧩 Modules
1. **Image Data Collection**  
2. **Preprocessing and Feature Extraction**  
3. **Model Implementation (SVM & KNN)**  
4. **Performance Evaluation and Visualization**  
5. **Prediction Interface (Tkinter GUI)**  

---

## 🖼️ Output Screens
- Upload dataset and split into train/test sets  
- Execute SVM and KNN classification  
- Predict liver condition (Normal or Tumor)  
- Accuracy comparison graph  

---

## 📊 Results
| Algorithm | Accuracy | Remarks |
|------------|-----------|----------|
| SVM        | High Accuracy | Best for structured image data |
| KNN        | Moderate Accuracy | Works well but less efficient for large datasets |

---

## 🏁 Conclusion
This project demonstrates how **machine learning** can assist in **medical image diagnostics** by automating liver cirrhosis detection.  
The comparison of SVM and KNN highlights SVM’s superior accuracy and efficiency, making it ideal for real-world healthcare applications.

---

## 🚀 Future Scope
- Integrate **deep learning models (CNNs)** for higher accuracy.  
- Make the system **cross-platform and web-based**.  
- Extend to detect other **liver-related diseases**.  
- Integrate with **Electronic Health Records (EHR)** for real-time diagnosis.

---

## 👨‍💻 Contributor
- **T. Sai Teja** (22N31A67G8)  

---

## 📚 References
- World Health Organization (WHO): [Liver Cirrhosis Overview](https://www.who.int/newsroom/fact-sheets/detail/hepatitis-c)  
- Kaggle Dataset: [Indian Liver Patient Dataset (ILPD)](https://www.kaggle.com/datasets)  
- R. Stoean et al., *AI in Liver Fibrosis Detection*, *Artificial Intelligence in Medicine*, 2011.  
