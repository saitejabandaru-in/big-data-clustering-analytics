<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:4F7CAC,100:1C2B36&height=180&section=header&text=Big%20Data%20Clustering%20Analytics&fontSize=36&fontColor=E6EEF3&animation=fadeIn&fontAlignY=35&desc=Scalable%20Big%20Data%20Clustering%20for%20Urban%20Mobility%20and%20Fraud%20Detection&descAlignY=55" />
</p>

<p align="center">
  🚕 Urban Mobility &nbsp;&nbsp;|&nbsp;&nbsp; 💳 Fraud Detection &nbsp;&nbsp;|&nbsp;&nbsp; 🧠 Scalable Clustering
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue"/>
  <img src="https://img.shields.io/badge/ML-Unsupervised%20Learning-brightgreen"/>
  <img src="https://img.shields.io/badge/Big%20Data-Clustering-orange"/>
  <img src="https://img.shields.io/badge/License-MIT-success"/>
  <img src="https://img.shields.io/badge/Status-Active-blue"/>
</p>



## 🔍 What is this project?
This repository contains a **real-world, scalable clustering system** designed to discover patterns and anomalies in large, complex datasets.

The project focuses on two impactful domains:

- **Urban Mobility Analysis (NYC Taxi Trips)**
- **Credit Card Fraud Detection**

The goal is to show how **modern clustering algorithms** such as **KMeans++, DBSCAN, OPTICS, BIRCH, and DENCLUE** perform when applied to **big data and high-dimensional data** — the kind of problems faced in industry.

This is not a toy example — it is a **research-grade and production-inspired clustering framework**.

---

## 🧠 Why this matters
Real-world data is:
- Large
- Noisy
- High-dimensional
- Mostly unlabeled  

Traditional clustering methods break at this scale.  
This project demonstrates how **scalable and density-based algorithms** can uncover:

- Mobility patterns in a smart city
- Anomalous transactions in financial data
- Meaningful clusters without labels

---

## 📊 Datasets Used

This project uses two publicly available Kaggle datasets:

### 🚕 NYC Taxi Trip Duration Dataset
Used to analyze:
- High-demand routes
- Travel-time clusters
- Trip distance vs duration
- Urban movement behavior

Features:
- Pickup & drop-off coordinates  
- Trip distance (computed using Euclidean distance)  
- Trip duration  
- Passenger count  

---

### 💳 Credit Card Fraud Detection Dataset
A real-world financial dataset with:
- **284,807 transactions**
- **492 fraud cases (0.17%)**
- **28 PCA-transformed features**

Fraud is treated as an **anomaly detection problem**, where unusual transactions form sparse clusters.

---

## ⚙️ Algorithms Implemented

Each algorithm is implemented as a **separate Python file** for clarity and modularity.

| Category | Algorithms |
|--------|----------|
| Partition-based | Mini-Batch KMeans++, CLARA, CLARANS |
| Hierarchical | BIRCH, CURE |
| Density-based | DBSCAN, OPTICS, DENCLUE |
| Grid-based | STING |

This structure allows easy testing, comparison, and reuse.

---

## 🧩 Project Workflow

```
Raw Data (Kaggle)
   ↓
Cleaning & Feature Engineering
   ↓
Scaling & PCA (for fraud data)
   ↓
Individual Clustering Algorithms
   ↓
Validation Metrics
   ↓
Visualization & Insights
```

---

## 📈 Evaluation Metrics
Clustering quality is measured using:

- **Silhouette Score**
- **Davies–Bouldin Index**
- **Adjusted Rand Index (ARI)**
- **Entropy**

These evaluate how well clusters are separated, compact, and meaningful.

---

## 🏆 Key Findings
- **Mini-Batch KMeans++** scales efficiently for millions of taxi trips  
- **BIRCH** clusters big data with low memory usage  
- **OPTICS and DENCLUE** are highly effective for fraud detection  
- Density-based methods isolate fraudulent transactions as **anomalies**

This confirms why **hybrid clustering strategies** are needed in real-world analytics.

---

## 📥 Downloading the Data

Due to Kaggle licensing and file size limits, datasets are **not stored in this repository**.

Please download them from:

- NYC Taxi Trip Duration  
  https://www.kaggle.com/c/nyc-taxi-trip-duration  

- Credit Card Fraud Detection  
  https://www.kaggle.com/mlg-ulb/creditcardfraud  

After downloading, place the CSV files into:
```
data/raw/
```

---

## 🚀 How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

Run any clustering algorithm:
```bash
python kmeans.py
python dbscan.py
python optics.py
python denclue.py
python birch.py
python clara.py
python clarans.py
```

Each file runs the full pipeline for that specific algorithm.

---

## 🧠 What this project demonstrates
This project shows hands-on skills in:

- Big data preprocessing
- Scalable machine learning
- Unsupervised learning
- Anomaly detection
- Feature engineering
- Model evaluation
- High-dimensional data handling

These are core skills used in:
- FinTech
- Smart cities
- Risk analytics
- Data engineering
- AI research

---

## 👨‍💻 Author
**Sai Teja Bandaru**  
Bachelor’s in Data Analytics  
Università degli Studi della Campania Luigi Vanvitelli  

---

## ⭐ If you like this project
Feel free to star ⭐ the repository or use it as a reference for:
- Research
- Data science portfolios
- Machine learning engineering
- Big data analytics

This repository represents **real-world clustering at scale**.
