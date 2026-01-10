# 🚀 Scalable Big Data Clustering for Urban Mobility & Fraud Detection

## 🔍 What is this project?
This project is a **real-world, large-scale clustering system** built to extract hidden patterns and anomalies from massive datasets.  
It focuses on two high-impact domains:

- **Urban Mobility (NYC Taxi Trips)**
- **Financial Security (Credit Card Fraud Detection)**

The goal is to show how **modern clustering algorithms scale, behave, and perform** when applied to millions of data points and high-dimensional feature spaces.

This repository is not a toy demo — it is a **research-grade and industry-relevant clustering framework**.

---

## 🧠 Why this matters
In the real world, data is:
- Huge  
- Noisy  
- High-dimensional  
- Often unlabeled  

Traditional clustering methods struggle at this scale.

This project demonstrates how **scalable clustering algorithms** like **Mini-Batch KMeans, DBSCAN, OPTICS, BIRCH, and DENCLUE** can be used to:
- Discover mobility patterns in a smart city
- Detect fraud in highly imbalanced financial data
- Handle millions of records efficiently

---

## 📊 Datasets Used

### 🚕 NYC Taxi Trip Duration Dataset
Used to understand:
- High-demand routes
- Travel time clusters
- Distance vs duration patterns
- Urban mobility behavior

Features:
- Pickup & drop-off coordinates
- Trip distance (computed using Euclidean distance)
- Trip duration
- Passenger count

---

### 💳 Credit Card Fraud Detection Dataset
A real-world fraud dataset with:
- **284,807 transactions**
- **492 fraud cases (0.17%)**
- **28 PCA-transformed features**

Fraud is treated as an **anomaly detection problem** using density-based clustering.

---

## ⚙️ Algorithms Implemented

| Category | Algorithms |
|--------|----------|
| Partition-based | Mini-Batch KMeans++, CLARA, CLARANS |
| Hierarchical | BIRCH, CURE |
| Density-based | DBSCAN, OPTICS, DENCLUE |
| Grid-based | STING |

Each algorithm is implemented and evaluated using the same pipeline for fair comparison.

---

## 🧩 Pipeline

```
Raw Data
   ↓
Data Cleaning & Feature Engineering
   ↓
Scaling & PCA (for fraud data)
   ↓
Clustering Algorithms
   ↓
Validation Metrics
   ↓
Visualization & Insights
```

---

## 📈 Evaluation Metrics
To objectively compare clustering quality, we use:

- **Silhouette Score**
- **Davies–Bouldin Index**
- **Adjusted Rand Index (ARI)**
- **Entropy**

These measure:
- Cluster compactness
- Separation
- Stability
- Anomaly isolation

---

## 🏆 Key Results
- **Mini-Batch KMeans++** scales efficiently on millions of NYC taxi trips  
- **BIRCH** handles big data with very low memory usage  
- **OPTICS & DENCLUE** outperform KMeans for fraud detection  
- Density-based models isolate fraud as **sparse anomalous clusters**

This shows why **one algorithm is never enough** — real big data needs **hybrid clustering strategies**.

---

## 🚀 How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

Run clustering:
```bash
python src/kmeans.py
python src/dbscan.py
python src/optics.py
python src/denclue.py
python src/birch.py
```

---

## 🧠 What this project demonstrates
This project proves practical skills in:

- Big data preprocessing
- Scalable machine learning
- Unsupervised learning
- Anomaly detection
- Feature engineering
- Model evaluation
- High-dimensional data handling

It reflects how clustering is used in:
- Smart cities  
- Banking & fintech  
- Risk analytics  
- Behavioral modeling  

---

## 👨‍💻 Author
**Sai Teja Bandaru**  
Bachelor’s in Data Analytics  
Università degli Studi della Campania Luigi Vanvitelli  

---

## ⭐ If you like this project
Feel free to **star the repo**, fork it, or use it as a reference for:
- Research
- ML engineering
- Data science portfolios
- Big data analytics

This repository represents **real-world clustering at scale**.
