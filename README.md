# 📘 K-Nearest Neighbors (KNN)

## Introduction to KNN Algorithm

### Principle of KNN
- KNN is a **supervised learning algorithm**.
- It is used for **classification** and **regression**.
- Predicts based on:
  1. Calculating distances to all samples.
  2. Selecting the `k` nearest neighbors.
  3. Using majority vote (for classification) or average (for regression).

---

### KNN for Classification and Regression

#### Classification Process:
1. Decide value of `k`
2. Measure distance to all training samples
3. Select `k` closest ones
4. Use majority vote for prediction

#### Regression Process:
- Predict the value by averaging the `k` nearest neighbor values.

---

### Distance Metrics

**Euclidean**        :\
$\text{Distance}(A, B) = \sqrt{\sum_{i=1}^{n}(A_i - B_i)^2}$                                                                   

**Manhattan**        :\
$\text{Distance}(A, B) = \sum_{i=1}^{n} \left| A_i - B_i \right|$                                                                 

**Chebyshev**        :\
$\text{Distance}(A, B) = \max_{i=1}^{n} \left| A_i - B_i \right|$                                                              

**Minkowski**        :\
$\text{Distance}(A, B) = \left( \sum_{i=1}^{n} \left| A_i - B_i \right|^p \right)^{\frac{1}{p}}$                               

**Cosine Similarity** :\
$\text{Similarity}(A, B) = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \cdot \sqrt{\sum_{i=1}^{n} B_i^2}}$       

---

## 🔧 Common Parameters, Attributes, and Methods

### Parameters (in `sklearn.neighbors.KNeighborsClassifier` / `KNeighborsRegressor`)

| Parameter         | Description |
|------------------|-------------|
| `n_neighbors`      | Number of neighbors to use (K). |
| `weights`          | Weight function: `'uniform'` or `'distance'`. |
| `algorithm`        | Search algorithm: `'auto'`, `'ball_tree'`, `'kd_tree'`, `'brute'`. |
| `metric`           | Distance metric: `'minkowski'`, `'euclidean'`, `'manhattan'`, etc. |
| `p`                | Power parameter for Minkowski metric (1 = manhattan, 2 = euclidean). |

### Attributes (after fitting)

| Attribute          | Description |
|-------------------|-------------|
| `classes_`         | Class labels (for classifier). |
| `effective_metric_`| Actual distance metric used. |
| `n_features_in_`   | Number of input features. |

### Methods

| Method             | Description |
|-------------------|-------------|
| `fit(X, y)`        | Store training data. |
| `predict(X)`       | Predict class or value for test data. |
| `predict_proba(X)` | Predict class probabilities (classifier). |
| `kneighbors(X)`    | Find K-nearest neighbors for `X`. |
| `score(X, y)`      | Return accuracy or \( R^2 \) score. |

---

## 📏 Evaluation Metrics

### For Classification

| Metric       | Description |
|--------------|-------------|
| Accuracy     | Proportion of correct predictions. |
| Precision    | How many predicted positives were correct. |
| Recall       | How many actual positives were correctly predicted. |
| F1 Score     | Harmonic mean of precision and recall. |
| Confusion Matrix | Shows true vs predicted class counts. |

### For Regression

| Metric       | Description |
|--------------|-------------|
| Mean Squared Error (MSE) | Average squared prediction error. |
| Mean Absolute Error (MAE) | Average absolute error. |
| \( R^2 \) Score           | Proportion of variance explained. |

---

### Comparison: KNN vs. K-means

| Aspect        | KNN                                   | K-means                                       |
|---------------|----------------------------------------|-----------------------------------------------|
| Learning Type | Supervised                            | Unsupervised                                  |
| Goal          | Predict label/value                   | Group data into `k` clusters                  |
| Meaning of k  | Number of neighbors to consider       | Number of clusters                            |

---

## Practical Case: Wine Type Classification

### Dataset Description
- 178 wine samples
- 13 features (e.g., Alcohol, Flavanoids, etc.)
- Goal: Classify into one of 3 classes (Class 0, 1, 2)

[Demo Code](/notebooks/knn.ipynb)
