# Decision Tree Regressor

> A supervised machine learning algorithm for predicting continuous numerical values using a tree-based decision structure.

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
| **Task** | Regression |
| **Library** | `sklearn.tree.DecisionTreeRegressor` |

A **Decision Tree Regressor** predicts continuous target values by recursively partitioning the feature space based on optimal thresholds. The algorithm constructs a hierarchical tree structure where each internal node represents a decision rule, and each leaf node outputs the **mean value** of training samples within that partition.

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
| **Input (X)** | Numerical feature matrix *(categorical features must be encoded)* |
| **Output (y)** | Continuous numerical values |

---

## Data Preprocessing

| Preprocessing Step | Required | Notes |
|--------------------|----------|-------|
| Feature Scaling | ❌ No | Tree-based models are scale-invariant |
| Missing Value Handling | ✅ Yes | Impute or remove missing values |
| Categorical Encoding | ✅ Yes | Use `OneHotEncoder` or `LabelEncoder` |
| Outlier Treatment | ⚠️ Optional | Trees are generally robust, but extreme values may affect splits |

---

## Algorithm Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Initialize with complete dataset at root node           │
│                           ↓                                 │
│  2. Evaluate all possible feature splits                    │
│                           ↓                                 │
│  3. Select split that minimizes variance (MSE criterion)    │
│                           ↓                                 │
│  4. Recursively partition until stopping criteria met       │
│                           ↓                                 │
│  5. Assign mean target value to each leaf node              │
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
| `criterion` | str | `squared_error` | Split quality measure (`squared_error`, `friedman_mse`, `absolute_error`) |

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
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Average absolute prediction error |
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Average squared error *(penalizes large errors)* |
| **RMSE** | $\sqrt{MSE}$ | Root mean squared error *(same units as target)* |
| **R² Score** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained *(0 to 1)* |

---

## Pros & Cons

| ✅ Advantages | ❌ Disadvantages |
|---------------|------------------|
| Captures non-linear relationships | Prone to overfitting without regularization |
| No feature scaling required | High variance *(sensitive to data changes)* |
| Highly interpretable & visualizable | Poor extrapolation beyond training range |
| Handles feature interactions naturally | Single trees may underperform on complex data |

---

## Implementation Example

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model initialization with regularization
model = DecisionTreeRegressor(
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
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE:  {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
```

---

<div align="center">

**📚 Related:** [Decision Tree Classifier](../Classifier/) | [Random Forest Regressor](#) | [Gradient Boosting](#)

</div>