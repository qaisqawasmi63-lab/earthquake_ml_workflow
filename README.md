# Earthquake Magnitude — Machine Learning Workflow

A full machine learning workflow applied to the USGS Significant Earthquakes dataset (1965–2016), covering data cleaning, feature engineering, clustering, classification, and regression.


## Dataset

`earthquake.csv` — 23,412 significant earthquake records (magnitude ≥ 5.5) with location, depth, magnitude, and time. Source: [USGS / Kaggle "Significant Earthquakes, 1965-2016"](https://www.kaggle.com/datasets/usgs/earthquake-database).

## Workflow

| Order | Notebook | What it does |
|---|---|---|
| 1 | `1_Cleaning_Stage.ipynb` | Filters to actual earthquakes (drops nuclear explosions/rock bursts), combines Date+Time, drops columns >50% missing, drops rows missing essentials |
| 2 | `2_Feature_Engineering_Stage.ipynb` | Extracts Year/Month/DayOfWeek, computes a 30-day rolling average magnitude, bins Magnitude into Minor/Moderate/Major categories |
| 3 | `3_Clustering_Analysis_Stage.ipynb` | Compares K-Means vs. DBSCAN on geographic location |
| 4 | `4_Classification_Analysis_Stage.ipynb` | Predicts magnitude category (Minor/Moderate/Major) |
| 5 | `5_Regression_Analysis_Stage.ipynb` | Predicts exact magnitude value |

Notebooks 3, 4, and 5 all branch independently off `earthquake_engineered.csv` (stage 2's output) — run 1 → 2 in order, then 3/4/5 in any order.

## Results

### Clustering (geographic grouping)

| Method | Clusters found | Noise points | Silhouette score |
|---|---|---|---|
| K-Means (k=6) | 6 (fixed) | — | 0.575 |
| DBSCAN | 4 (data-driven) | 27 (0.1%) | 0.070 |

K-Means scores higher on silhouette, but that's partly an artifact of the metric favoring compact circular clusters — which is exactly the shape K-Means is built to produce. DBSCAN's lower score comes from following the actual, irregular geometry of fault lines instead of forcing earthquakes into circular groups. Visually comparing `cluster_comparison.png` is more informative here than the silhouette number alone.

### Classification (predicting Minor / Moderate / Major)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Random Forest | 0.680 | 0.381 | 0.355 | 0.332 |
| Gradient Boosting | 0.694 | 0.509 | 0.363 | 0.341 |
| Extra Trees | 0.663 | 0.359 | 0.348 | 0.328 |
| SVC | 0.435 | 0.367 | 0.399 | 0.322 |

**Important limitation, stated plainly:** these scores look mediocre, and that's expected, not a bug. "Major" earthquakes (magnitude ≥ 7.0) are only ~3% of the data, so macro-averaged metrics (which weight all three classes equally) come out low even though overall accuracy looks decent. `class_weight='balanced'` was used to stop models from just predicting "Minor" for everything, but the underlying class imbalance is still the main limiting factor here — worth naming directly if asked about it.

### Regression (predicting exact magnitude)

| Model | MSE | RMSE | R² |
|---|---|---|---|
| Random Forest Regressor | 0.189 | 0.435 | 0.005 |
| Gradient Boosting Regressor | 0.177 | 0.421 | 0.066 |
| Support Vector Regression | 0.200 | 0.447 | -0.053 |

R² is very low across all three models — and that's a real, honest finding, not a failure of implementation. Earthquake magnitude is governed by fault mechanics (stress accumulation, rupture length, rock properties) that location, depth, and time alone don't capture. **State this as a known limitation in any write-up** rather than treating it as something to fix — an interviewer will respect "I found this doesn't predict well, and here's the physical reason why" far more than an inflated result.

## Tech Stack

- Python, pandas, numpy
- scikit-learn (KMeans, DBSCAN, RandomForest, GradientBoosting, ExtraTrees, SVC, SVR)
- matplotlib

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib
```

Run notebooks 1 and 2 in order, then 3, 4, 5 in any order.

## Files

- `earthquake.csv` — raw dataset
- `earthquake_preprocessed.csv`, `earthquake_engineered.csv`, `earthquake_clustered.csv` — intermediate outputs
- `classification_results.csv`, `regression_results.csv` — model comparison metrics
- `cluster_comparison.png` — side-by-side K-Means vs. DBSCAN visualization
