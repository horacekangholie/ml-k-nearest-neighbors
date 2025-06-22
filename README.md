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

| Metric               | Formula                                                  |
|----------------------|-----------------------------------------------------------|
| **Euclidean**        | `sqrt(sum((Ai - Bi)^2))`                                  |
| **Manhattan**        | `sum(|Ai - Bi|)`                                          |
| **Chebyshev**        | `max(|Ai - Bi|)`                                          |
| **Minkowski**        | `(sum(|Ai - Bi|^p))^(1/p)`                                |
| **Cosine Similarity**| `(sum(Ai * Bi)) / (sqrt(sum(Ai^2)) * sqrt(sum(Bi^2)))`    |

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
