# Decision Tree Classifier

> A supervised machine learning algorithm for classifying categorical outcomes using a tree-based decision structure.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Use Cases](#use-cases)
3. [Input & Output](#input--output)
4. [Data Preprocessing](#data-preprocessing)
5. [Algorithm Workflow](#algorithm-workflow)
6. [Hyperparameters](#hyperparameters)
7. [Regularization Techniques](#regularization-techniques)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Pros & Cons](#pros--cons)
10. [Implementation Example](#implementation-example)

---

## Overview

| Attribute | Description |
|-----------|-------------|
| **Type** | Supervised Learning |
| **Task** | Classification |
| **Library** | `sklearn.tree.DecisionTreeClassifier` |

A **Decision Tree Classifier** predicts categorical class labels by recursively partitioning the feature space based on optimal splits. The algorithm builds a hierarchical tree where each internal node represents a decision rule, and each leaf node assigns a class label based on the majority of samples in that partition.

---

## Use Cases

| Scenario | Suitability |
|----------|-------------|
| Non-linear feature-target relationships | ✅ Excellent |
| Complex feature interactions | ✅ Excellent |
| Model interpretability required | ✅ Excellent |
| Small to medium datasets | ✅ Good |
| Large-scale datasets | ⚠️ Consider ensemble methods |

---

## Input & Output

| Component | Description |
|-----------|-------------|
| **Input (X)** | Numerical/categorical feature matrix *(categorical features must be encoded)* |
| **Output (y)** | Discrete class labels |

---

## Data Preprocessing

| Preprocessing Step | Required | Notes |
|--------------------|----------|-------|
| Feature Scaling | ❌ No | Tree-based models are scale-invariant |
| Missing Value Handling | ✅ Yes | Impute or remove missing values |
| Categorical Encoding | ✅ Yes | Use `OneHotEncoder` or `LabelEncoder` |
| Outlier Treatment | ⚠️ Optional | Trees are robust, but extreme values may affect splits |

---

## Algorithm Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Initialize with complete dataset at root node           │
│                           ↓                                 │
│  2. Evaluate all possible feature splits                    │
│                           ↓                                 │
│  3. Select split that maximizes information gain            │
│     (e.g., Gini impurity or entropy)                        │
│                           ↓                                 │
│  4. Recursively partition until stopping criteria met       │
│                           ↓                                 │
│  5. Assign majority class label to each leaf node           │
└─────────────────────────────────────────────────────────────┘
```

---

## Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | int | None | Maximum tree depth *(primary regularization)* |
| `min_samples_split` | int | 2 | Minimum samples required to split an internal node |
| `min_samples_leaf` | int | 1 | Minimum samples required at a leaf node |
| `max_features` | int/float | None | Number of features to consider per split |
| `criterion` | str | `gini` | Split quality measure (`gini`, `entropy`, `log_loss`) |

---

## Regularization Techniques

To prevent overfitting, apply the following constraints:

| Technique | Implementation |
|-----------|----------------|
| Limit tree depth | Set `max_depth` to a reasonable value (e.g., 5–10) |
| Increase leaf samples | Set `min_samples_leaf` ≥ 5 |
| Increase split samples | Set `min_samples_split` ≥ 10 |
| Pruning | Use `ccp_alpha` for cost-complexity pruning |

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | $\frac{\text{Correct}}{\text{Total}}$ | Proportion of correct predictions |
| **Precision** | $\frac{TP}{TP+FP}$ | Correct positive predictions |
| **Recall** | $\frac{TP}{TP+FN}$ | Coverage of actual positives |
| **F1 Score** | $2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$ | Harmonic mean of precision and recall |
| **ROC-AUC** | - | Area under ROC curve (binary/multiclass) |

---

## Pros & Cons

| ✅ Advantages | ❌ Disadvantages |
|---------------|------------------|
| Captures non-linear relationships | Prone to overfitting without regularization |
| No feature scaling required | High variance *(sensitive to data changes)* |
| Highly interpretable & visualizable | Can create biased trees if classes are imbalanced |
| Handles feature interactions naturally | Single trees may underperform on complex data |

---

## Implementation Example

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

# Model initialization with regularization
model = DecisionTreeClassifier(
	max_depth=5,
	min_samples_leaf=5,
	min_samples_split=10,
	random_state=42
)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
# For binary/multiclass ROC-AUC (if applicable):
# print(f"ROC-AUC:   {roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr'):.4f}")
```

---

<div align="center">

**📚 Related:** [Decision Tree Regressor](../Regressor/) | [Random Forest Classifier](#) | [Gradient Boosting](#)

</div>
