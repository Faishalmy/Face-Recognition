# Face Recognition with MTCNN, Wavelet Transform, and SVM

This project is a facial recognition system built using Python and machine learning. It leverages **MTCNN** for face detection, combines raw and wavelet-transformed image features, and uses a **Support Vector Machine (SVM)** classifier with hyperparameter optimization via **GridSearchCV**.

## 🔍 Project Overview

The system performs the following steps:

1. **Face Detection**  
   Detect faces from a collection of images using the MTCNN Multi-task Cascaded Convolutional Networks detector.

2. **Preprocessing and Feature Extraction**
   - Detected faces are resized to **32x32 pixels**.
   - Each image is transformed using **Discrete Wavelet Transform (DWT)** to capture high-frequency details.
   - The raw and wavelet-transformed images are flattened and concatenated as input features.

3. **Dataset Preparation**
   - Each person has a folder with their face images.
   - Folder names are used to assign labels.
   - The combined feature set is split into training and test sets.

4. **Model Training with SVM**
   - Model used: `sklearn.svm.SVC`
   - Optimized using `GridSearchCV` with parameters:
     ```python
     {
         'svc__C': [1, 10, 100, 1000],
         'svc__kernel': ['linear', 'rbf']
     }
     ```

5. **Evaluation**
   - Accuracy, precision, recall, and F1-score are reported.
   - Confusion matrix is visualized with seaborn.

## 🧪 Model Evaluation Results

- **Best Parameters:** `{'svc__C': 1, 'svc__kernel': 'linear'}`
- **Best Cross-Validation Score:** `0.781`
- **Test Accuracy:** `0.78`

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.94      | 0.88   | 0.91     | 17      |
| 1     | 0.71      | 0.63   | 0.67     | 19      |
| 2     | 0.75      | 0.71   | 0.73     | 21      |
| 3     | 0.84      | 0.81   | 0.82     | 26      |
| 4     | 0.68      | 0.88   | 0.77     | 17      |
|       |           |        |          |         |
| **Overall Accuracy** |       |        | **0.78** |         |



