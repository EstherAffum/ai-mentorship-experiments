# ai-mentorship-experiments

> Machine learning experiments built during a structured AI Mentorship Programme — income prediction using supervised and unsupervised learning on the UCI Adult Census dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Model 1: Logistic Regression](#model-1-logistic-regression)
- [Model 2: KMeans Clustering](#model-2-kmeans-clustering)
- [Model Comparison](#model-comparison)
- [Key Insight](#key-insight)
- [How to Run](#how-to-run)
- [About](#about)

---

## Project Overview

This repository explores two machine learning approaches applied to the same dataset — one supervised, one unsupervised. The goal is to understand how each model works, what it produces, and where each approach is most appropriate.

Both notebooks are written with full step-by-step markdown explanations at every stage, making them suitable as learning references as well as working experiments.

---

## Dataset

**Name:** UCI Adult / Census Income Dataset  
**Records:** ~32,000  
**Features:** 14 (age, education, occupation, marital status, hours worked per week, capital gain, and others)  
**Target variable:** `label` — whether a person earns `<=50K` or `>50K` per year  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)

> **Note on this file's format:** Each row is wrapped in double-quotes. The notebooks handle this with a custom parsing step before loading into pandas.

---

## Repository Structure

```
ai-mentorship-experiments/
│
├── notebooks/
│   ├── income_prediction_classifier.ipynb   ← Logistic Regression (supervised)
│   └── income_kmeans.ipynb                  ← KMeans Clustering (unsupervised)
│
├── data/
│   └── income_prediction.csv
│
└── README.md
```

---

## Model 1: Logistic Regression

**Notebook:** `notebooks/income_prediction_classifier.ipynb`  
**Type:** Supervised Classification  
**Algorithm:** `sklearn.linear_model.LogisticRegression`

### What it does

Logistic Regression is a classification algorithm that learns the relationship between input features and a binary outcome. It computes a weighted sum of all 14 features, passes it through a sigmoid function to produce a probability, and predicts `>50K` if that probability exceeds 0.5.

### Pipeline

| Step | Action |
|------|--------|
| 1 | Load and parse the CSV |
| 2 | Exploratory data analysis — class distribution, feature histograms |
| 3 | Replace `?` placeholders with NaN and drop incomplete rows |
| 4 | Label-encode all categorical columns |
| 5 | Define features (X) and target (y) |
| 6 | Scale with `StandardScaler` — resolves ConvergenceWarning |
| 7 | 80/20 train-test split |
| 8 | Fit `LogisticRegression(max_iter=1000)` |
| 9 | Evaluate on the held-out test set |

### Why StandardScaler matters here

The dataset contains features with very different numeric ranges — `fnlwgt` runs into the hundreds of thousands while `sex` is simply 0 or 1. Without scaling, the solver takes tiny steps to compensate for the magnitude gap and hits its iteration limit before converging. Scaling all features to mean = 0, standard deviation = 1 fixes this entirely. The model converged in 11 iterations after scaling was applied.

### Results

| Metric | Value |
|--------|-------|
| Accuracy | ~85% |
| Precision (>50K class) | ~73% |
| Recall (>50K class) | ~59% |
| F1 Score (>50K class) | ~65% |
| ROC-AUC | ~0.90 |

### Interpretation

The model performs well on the majority class (`<=50K`) and reasonably on the minority class (`>50K`). The gap between precision and recall for the `>50K` class reflects the class imbalance in the dataset — roughly 75% of records are `<=50K`. The AUC of 0.90 indicates strong overall discriminative ability.

---

## Model 2: KMeans Clustering

**Notebook:** `notebooks/income_kmeans.ipynb`  
**Type:** Unsupervised Learning  
**Algorithm:** `sklearn.cluster.KMeans`

### What it does

KMeans is a clustering algorithm that groups data points into k clusters based on similarity, without using any labels during training. It assigns every person in the dataset to a cluster by minimising the distance between each point and its nearest cluster centre (centroid).

This experiment asks: can the model discover the income groupings from the features alone, without ever being told what the labels are?

### Pipeline

| Step | Action |
|------|--------|
| 1 | Same preprocessing pipeline as the classifier |
| 2 | Scale with `StandardScaler` — Euclidean distance is sensitive to magnitude |
| 3 | Elbow Method — plot inertia for k = 1 to 10 to find the optimal k |
| 4 | Silhouette scores — numeric measure of cluster separation quality |
| 5 | Fit `KMeans(n_clusters=2, random_state=42, n_init=10)` |
| 6 | PCA projection to 2D for cluster visualisation |
| 7 | Profile each cluster by mean feature values |
| 8 | Cross-tabulate cluster assignments against the true income labels |
| 9 | Calculate cluster purity |

### Why k = 2

Both the Elbow Method and Silhouette scores point to k = 2 as the natural choice. This also aligns with the problem domain — the original classification is binary. Choosing k = 2 lets us directly compare whether the unsupervised clusters correspond to the real income groups.

### Why scaling is critical here

KMeans measures Euclidean distance. A feature on a scale of 0 to 1,000,000 will completely dominate the distance calculation and make all other features irrelevant. StandardScaler puts every feature on the same scale so each one contributes equally to the clustering.

### Results

| Metric | Value |
|--------|-------|
| Optimal k (Elbow + Silhouette) | 2 |
| Cluster 0 purity | ~77% |
| Cluster 1 purity | ~76% |
| PCA variance captured (2 components) | ~30-35% |

### Key finding

With k = 2, KMeans recovers approximately 75-78% cluster purity against the actual income labels — without ever seeing them during training. The cluster that maps to `>50K` earners tends to have higher mean values for `education-num`, `capital-gain`, and `hours-per-week`. This confirms that the demographic features contain real signal about income group membership.

---

## Model Comparison

| | Logistic Regression | KMeans Clustering |
|---|---|---|
| Learning type | Supervised | Unsupervised |
| Uses income labels during training | Yes | No |
| Primary output | Predicted class (`<=50K` / `>50K`) | Cluster ID (0 or 1) |
| Accuracy on test set | ~85% | Not directly comparable |
| ROC-AUC | ~0.90 | N/A |
| Evaluation metrics | Accuracy, precision, recall, F1, AUC | Inertia, silhouette score, cluster purity |
| Best use case | Predicting a known labelled outcome | Discovering hidden structure in unlabelled data |

### Which model performed best?

**Logistic Regression is the stronger model for this specific problem.**

Because it is trained on the actual labels, it learns the exact decision boundary between the two income groups and can be held to a precise accuracy standard. An AUC of 0.90 places it well above the 0.5 random baseline.

KMeans cannot be judged on the same scale because it is solving a different problem. It never sees the labels. The fact that it achieves ~76% cluster purity is genuinely meaningful — it means the features alone carry enough signal to partially separate income groups — but it is not a prediction tool and should not be evaluated as one.

**Rule of thumb:**
- Use Logistic Regression when you have labelled data and want to predict a specific outcome.
- Use KMeans when you have no labels and want to understand what natural groupings exist in your data.

---

## Key Insight

Traditional data analysis describes patterns a human analyst already knows to look for. Machine learning finds patterns across many variables simultaneously and generalises them to new, unseen data.

A pivot table can show average income by education level. A trained Logistic Regression model can take a person it has never seen, consider all 14 features at once, and estimate their income group in milliseconds. KMeans goes further — it finds groupings without any human hypothesis about what those groups should be.

The critical difference is not speed. It is **generalisation** and **the ability to find structure that was never specified in advance**.

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/ai-mentorship-experiments.git
cd ai-mentorship-experiments

# Install dependencies
pip install scikit-learn pandas matplotlib seaborn jupyter

# Add the dataset to the data/ folder
# income_prediction.csv — place it there before running

# Launch Jupyter
jupyter notebook notebooks/
```

Open either notebook and run all cells from top to bottom. The income_prediction.csv file must be in the same directory as the notebook, or update the file path in the loading cell.

---

## About

Built by **Esther Dankwah Affum**  
BSc Mathematics — Kwame Nkrumah University of Science and Technology (KNUST)  
Data Analyst | Accra, Ghana  
AI Mentorship Programme — Agentic AI, Automation, and Machine Learning track

Areas of interest: machine learning for financial inclusion, alternative credit scoring models, and data-driven decision-making in the Ghanaian banking sector.

---

*Python 3.10 | scikit-learn 1.x | Notebooks executed and verified end-to-end*
