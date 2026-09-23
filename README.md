# 🤖 Tata Tech Pulse — Lab Assignments

> **Course:** Applied AI ML | **Track:** AI & ML | **Level:** Intermediate
> **Program:** TechPulse FY-26 — Tata Technologies Ltd.

---

## 📋 Overview

This repository contains **10 hands-on lab assignments** covering the full spectrum of Applied AI & Machine Learning — from classical ML and data preprocessing to deep learning, computer vision, NLP, and MLOps.

Each lab is self-contained in its own folder with:
- ✅ Jupyter notebook (`.ipynb`) with the full, runnable code
- ✅ Individual `README.md` with objectives, methodology, and expected output
- ✅ Synthetic / self-generated datasets (no manual downloads needed for any lab)

---

## 📂 Repository Structure

```
assignments/
├── README.md                              ← You are here
├── requirements.txt                       ← All Python dependencies
│
├── Lab01_Car_Mileage_Estimation/          ← Linear Regression & Random Forest on MPG
├── Lab02_Simulated_Driving_Agent/         ← Genetic Algorithm driving agent
├── Lab03_Data_Cleaning_Preprocessing/     ← Missing values, outliers, encoding, scaling
├── Lab04_Vehicle_Price_Prediction/        ← Pipeline: Linear/Random Forest/Gradient Boosting
├── Lab05_Predictive_Maintenance/          ← Sensor-based failure classification
├── Lab06_Traffic_Sign_CNN/                ← CNN on simulated GTSRB-style signs
├── Lab07_Pedestrian_Detection/            ← HOG + SVM with OpenCV
├── Lab08_Sentiment_Analysis_LSTM/         ← LSTM sentiment classifier on vehicle feedback
├── Lab09_Feature_Importance/              ← Impurity / Permutation / SHAP importance
└── Lab10_MLOps_Workflow/                  ← MLflow + Docker + CI/CD pipeline
```

---

## 🧪 Lab Assignments

| # | Lab Title | Key Concepts | Notebook |
|:-:|-----------|-------------|----------|
| 1 | **Car Mileage Estimation** | Linear Regression, Random Forest, Evaluation Metrics (MAE/RMSE/R²) | `TTL_Assignment_01.ipynb` |
| 2 | **Simulated Driving Agent** | Genetic Algorithms, Fitness/Crossover/Mutation, Agent-based Simulation | `TTL_Assignment_02.ipynb` |
| 3 | **Data Cleaning & Preprocessing** | Missing Value Imputation, IQR Outlier Capping, One-Hot Encoding, Scaling | `TTL_Assignment_03.ipynb` |
| 4 | **Vehicle Price Prediction** | ColumnTransformer Pipelines, Linear/Random Forest/Gradient Boosting comparison | `TTL_Assignment_04.ipynb` |
| 5 | **Predictive Maintenance** | Classification, Class Imbalance, Confusion Matrix, Stratified Split | `TTL_Assignment_05.ipynb` |
| 6 | **Traffic Sign Classification (CNN)** | CNNs, Image Preprocessing, Multi-class Classification | `TTL_Assignment_06.ipynb` |
| 7 | **Pedestrian Detection** | HOG Descriptors, Pre-trained SVM, OpenCV `detectMultiScale` | `TTL_Assignment_07.ipynb` |
| 8 | **Sentiment Analysis (LSTM)** | Tokenization, Embeddings, LSTM, Binary Text Classification | `TTL_Assignment_08.ipynb` |
| 9 | **Feature Importance Visualization** | Impurity-based Importance, Permutation Importance, SHAP | `TTL_Assignment_09.ipynb` |
| 10 | **MLOps Workflow Simulation** | MLflow Tracking & Model Registry, Docker, CI/CD (GitHub Actions) | `TTL_Assignment_10.ipynb` |

---

## 🛠️ Prerequisites

- Python 3.8 or higher
- Basic programming skills in Python
- Familiarity with data structures and algorithms
- Introductory understanding of ML/DL concepts
- Comfort with using Jupyter Notebook / terminal / command line

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/DvitGohil/Applied-AI-ML-Labs.git
cd Applied-AI-ML-Labs
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

Each lab is a self-contained Jupyter notebook and can be run independently, top to bottom:

```bash
# Launch Jupyter
jupyter notebook

# Then open and run any lab notebook, e.g.:
# Lab01_Car_Mileage_Estimation/TTL_Assignment_01.ipynb
# Lab05_Predictive_Maintenance/TTL_Assignment_05.ipynb
```

All datasets are generated in-notebook (`np.random.seed(42)` / `random.seed(42)`), so no manual dataset download is required to run any lab end to end.

> **Note:** Lab 6, 7, and 8 use **synthetic stand-in data** (simulated traffic sign images, a synthetic street scene, and templated feedback text respectively) since real datasets (GTSRB, street photos, real reviews) were not available in the build environment. The modeling code is fully correct and production-ready — swap in the real dataset referenced in each lab's own README for real-world results.
>
> Lab 10 additionally requires a local **Docker** installation and **MLflow** to fully execute the containerization and CI/CD steps — refer to `Lab10_MLOps_Workflow/README.md` for details.

---

## 📚 Software Requirements

| Software | Version |
|----------|---------|
| Python | 3.8+ |
| NumPy | ≥ 1.24.0 |
| Pandas | ≥ 2.0.0 |
| Scikit-learn | ≥ 1.3.0 |
| Matplotlib | ≥ 3.7.0 |
| Seaborn | ≥ 0.12.0 |
| TensorFlow / Keras | ≥ 2.13.0 |
| OpenCV | ≥ 4.8.0 |
| SHAP | ≥ 0.44.0 |
| MLflow | ≥ 2.5.0 |
| Docker | Latest |

See `requirements.txt` for the complete pinned dependency list.

---

## 📖 Recommended Books

- *Introduction to AI & Machine Learning* — Munesh Chandra Trivedi & Ankit Srivastava
- *Python Machine Learning* — Sebastian Raschka & Vahid Mirjalili
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* — Aurélien Géron

---

## 🙋 Author

**Sumit Deshpande**
