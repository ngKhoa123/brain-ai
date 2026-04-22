# 🧠 Brain Tumor Classification AI (Deep Learning vs SVM)

A full-stack AI web application for brain tumor classification from MRI images, comparing:

*  Deep Learning (EfficientNet-B0)
*  Machine Learning (SVM + HOG)

---

## Live Demo

https://your-render-link.onrender.com

---

## 📌 Overview

This project classifies brain MRI images into 4 categories:

* Glioma
* Meningioma
* Pituitary tumor
* No tumor

Users can upload an MRI image and get predictions from two different approaches, enabling direct comparison.

This project follows an end-to-end pipeline:

- Data → Model → Web
---

## ⚠️ Disclaimer

 ⚠️ This AI model is trained on the BRISC2025 Kaggle dataset.
 It is for educational/demo purposes only and
 **does NOT replace professional medical diagnosis**.

---

##  Dataset

* Source: https://www.kaggle.com/datasets/briscdataset/brisc2025
* Classes: 4 tumor types
* Train samples: 5000 images
* Test samples: 1000 images

---

##  Model 1: Deep Learning (EfficientNet-B0)

* Framework: PyTorch
* Architecture: EfficientNet-B0
* Input: MRI grayscale → converted to 3 channels

###  Classification Report

```
              precision    recall  f1-score   support

      glioma       0.99      0.98      0.99       254
  meningioma       0.99      0.98      0.99       306
    no_tumor       1.00      1.00      1.00       140
   pituitary       0.98      1.00      0.99       300

    accuracy                           0.99      1000
   macro avg       0.99      0.99      0.99      1000
weighted avg       0.99      0.99      0.99      1000
```

---

##  Model 2: Machine Learning (SVM + HOG)

* Feature extraction: HOG (Histogram of Oriented Gradients)
* Classifier: Support Vector Machine (SVM)
* Framework: Scikit-learn

###  Classification Report

```text
              precision    recall  f1-score   support

      glioma       0.96      0.90      0.93       254
  meningioma       0.92      0.92      0.92       306
    no_tumor       0.97      1.00      0.99       140
   pituitary       0.96      1.00      0.98       300

    accuracy                           0.95      1000
   macro avg       0.95      0.96      0.95      1000
weighted avg       0.95      0.95      0.95      1000
```

### Comparison with Deep Learning

| Model           | Accuracy |
| --------------- | -------- |
| EfficientNet-B0 | **0.99** |
| SVM + HOG       | 0.95     |

###  Insights

* Deep Learning achieves higher accuracy and better generalization
* SVM performs well but struggles more on complex tumor patterns
* HOG features are limited compared to learned deep features

This demonstrates the advantage of deep learning for medical imaging tasks.

## Web Application

* Backend: Flask
* Frontend: HTML + CSS + JavaScript
* Deployment: Render

### Features

* Upload MRI image
* Predict using both models
* Display probability distribution
* Compare DL vs SVM results

---

## Training Notebook

Training process is available in:

```
notebooks/BRISC.ipynb
```

Includes:

* Data preprocessing
* Model training
* Evaluation

---

##  Project Structure

```
brain-ai/
│
├── app.py
├── preprocess.py
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── notebooks/
│   └── BRISC.ipynb
│
└── uploads/
```

---

##  Installation (Local)

```bash
git clone https://github.com/ngKhoa123/brain-ai.git
cd brain-ai

pip install -r requirements.txt
python app.py
```

---

## Deployment (Render)

* Models are NOT stored in GitHub (due to size limits)
* Models are downloaded dynamically using `gdown` from Google Drive

---

## Key Learnings

* Handling large ML models in deployment
* Comparing Deep Learning vs Traditional ML
* Building full-stack AI applications
* Deploying AI systems

---

##  Author

**Khoa Nguyen**
##  Future Improvements

* **Collect more diverse data**

  * Expand dataset with more MRI images from different sources
  * Include variations in scanners, hospitals, and patient demographics
  * Improve model generalization and reduce bias

*  **Enhance model performance**

  * Experiment with advanced architectures (EfficientNetV2, Vision Transformers)
  * Fine-tune hyperparameters and training strategies

*  **Model explainability**

  * Integrate Grad-CAM to visualize tumor regions
  * Improve trust and interpretability for users

*  **Optimize deployment**

  * Reduce model size and inference time
  * Enable faster real-time predictions

* **Production readiness**

  * Build REST API for integration
  * Add logging, monitoring, and error handling

---




