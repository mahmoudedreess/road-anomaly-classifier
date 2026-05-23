# RoadGuard CV — Milestone 1: HOG \+ SVM Road Hazard Classifier

**Course Project — Berlin German International University** **City of Berlin, Urban Infrastructure & Smart Mobility Division — 2026**

---

## Project Overview

Every year, thousands of vehicles are damaged due to undetected road hazards. Manual road inspection covers less than 15% of Berlin's 5,400 km road network per inspection cycle — it is expensive, slow, and reactive.

**RoadGuard** is an AI-powered road hazard detection system designed to be integrated into dashcam-equipped municipal vehicles and smart road cameras. As vehicles drive their normal daily routes, the system continuously analyzes the road surface and flags hazards in real time.

**Milestone 1** establishes the baseline: a reproducible pipeline that collects real road hazard imagery from the web and trains a HOG \+ SVM classifier to distinguish between potholes and road debris. This classifier serves as the performance benchmark against which deep learning models in later milestones will be evaluated.

---

## Problem Definition

| Item | Detail |
| :---- | :---- |
| Task | Binary image classification |
| Class 0 | Pothole — Depression / crater in road surface |
| Class 1 | Debris / Object — Rocks, branches, fallen objects on road |
| Input | Single road-surface image (128×128 px, grayscale) |
| Output | Predicted class \+ confidence score (0.0–1.0) |
| Baseline method | HOG feature extraction \+ SVM classification |
| Train/test split | 80% training / 20% test (stratified) |
| Evaluation | Accuracy, F1 (weighted), Precision, Recall, ROC-AUC, CV F1 |

---

## Pipeline

iCrawler (Web Scraping) → Preprocessing → HOG → SVM → Evaluation

| Step | Description | Tool |
| :---- | :---- | :---- |
| 1 | Scrape road hazard images from the web | iCrawler (BingImageCrawler) |
| 2 | Manual curation to supplement scraped data | Google Colab upload |
| 3 | Validate dataset — check all images are readable | OpenCV |
| 4 | Preprocess: grayscale → CLAHE → resize 128×128 | OpenCV |
| 5 | Extract HOG features → 1,764-dimensional vector | scikit-image |
| 6 | Train RBF SVM classifier | scikit-learn |
| 7 | Evaluate: Accuracy, F1, ROC-AUC, CV F1, Confusion Matrix | scikit-learn |
| 8 | Save model, scaler, HOG config as .pkl files | joblib |

---

## HOG Parameters

| Parameter | Value | Rationale |
| :---- | :---- | :---- |
| orientations | 9 | Covers 0°–180° gradient directions |
| pixels\_per\_cell | (16, 16\) | Captures local texture patch |
| cells\_per\_block | (2, 2\) | 32×32 px normalization window |
| block\_norm | L2-Hys | Robust to gradient outliers |
| image size | 128×128 | Balance between detail and speed |
| **Feature vector size** | **1,764 dimensions** | 7×7 blocks × 4 cells × 9 orientations |

---

## Results

| Metric | Value |
| :---- | :---- |
| Test Accuracy | 82.35% |
| ROC-AUC | 0.8333 |
| CV F1 (5-fold) | 0.8291 ± 0.1140 |

---

## Project Structure

road-anomaly-classifier/

│

├── roadguard\_milestone1.ipynb   \# Main notebook

├── README.md                    \# This file

│

├── data/

│   ├── pothole\_on\_road/         \# Pothole images

│   └── road\_debris/             \# Debris images

│

└── models/

    ├── roadguard\_svm\_baseline.pkl  \# Trained SVM classifier

    ├── scaler.pkl                  \# StandardScaler

    └── hog\_config.pkl              \# HOG configuration

---

## How to Run

1. Open `roadguard_milestone1.ipynb` in Google Colab  
2. Run **Cell 1** — clear old data folders  
3. Run **Cell 2** — install imports  
4. Run **Cell 3** — iCrawler scraper (optional, commented out by default)  
5. Run **Cell 4** — upload your road hazard images manually  
6. Run **Cell 5** — validate dataset  
7. Run **Cell 6** — preprocess images and extract HOG features  
8. Run **Cell 7** — visualize HOG features for both classes  
9. Run **Cell 8** — train SVM and evaluate  
10. Run **Cell 9** — save trained model

## Dataset Notes

iCrawler (BingImageCrawler) was used as the primary image collection pipeline across multiple search queries per class to maximize diversity. Due to search engine noise returning irrelevant images, the dataset was supplemented with manually curated road hazard photographs to ensure quality.

All images are real road photographs collected from public web sources.

