# Random Forest Regressor

> A supervised machine learning algorithm for predicting continuous numerical values using an ensemble of decision trees that aggregates predictions to improve accuracy and reduce overfitting.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Use Cases](#use-cases)
3. [Input & Output](#input--output)
4. [Data Preprocessing](#data-preprocessing)
5. [Algorithm Workflow](#algorithm-workflow)
6. [Hyperparameters](#hyperparameters)
7. [Assumptions](#assumptions)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Pros & Cons](#pros--cons)
10. [Implementation Example](#implementation-example)

---

## Overview

| Attribute   | Description                                   |
|-------------|-----------------------------------------------|
| **Type**    | Supervised Learning                           |
| **Task**    | Regression                                    |
| **Library** | `sklearn.ensemble.RandomForestRegressor`      |

Random Forest Regressor is an ensemble learning method that constructs multiple decision trees during training and outputs the average prediction of individual trees for regression tasks. It uses bootstrap aggregating (bagging) and random feature selection to create diverse trees, reducing variance and preventing overfitting while maintaining low bias.

---

## Use Cases

| Scenario                        | Suitability      |
|----------------------------------|------------------|
| Non-linear feature-target relationship | ✅ Excellent    |
| High-dimensional datasets         | ✅ Excellent    |
| Feature importance analysis       | ✅ Excellent    |
| Handling outliers                 | ✅ Excellent    |
| Mixed data types (numerical + categorical) | ✅ Excellent    |
| Large-scale datasets              | ✅ Good         |
| Real-time predictions             | ⚠️ Moderate (slower than single trees) |
| Model interpretability            | ⚠️ Limited (black box) |

---

## Input & Output

| Component      | Description                                      |
|--------------- |--------------------------------------------------|
| **Input (X)**  | Numerical and categorical feature matrix *(categorical features should be encoded)* |
| **Output (y)** | Continuous numerical values                      |

---

## Data Preprocessing

| Preprocessing Step     | Required | Notes                                                      |
|-----------------------|----------|------------------------------------------------------------|
| Feature Scaling       | ❌ No    | Tree-based models are invariant to feature scaling         |
| Missing Value Handling| ⚠️ Optional | Can handle NaN values in some implementations             |
| Categorical Encoding  | ✅ Yes   | Use `LabelEncoder` or `OrdinalEncoder` for ordinal, `OneHotEncoder` for nominal |
| Outlier Treatment     | ❌ No    | Robust to outliers due to ensemble averaging               |

---

## Algorithm Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Create bootstrap samples (random sampling with          │
│     replacement) from training data                         │
│                           ↓                                 │
│  2. For each bootstrap sample, build a decision tree:       │
│     - At each node, randomly select subset of features      │
│     - Choose best split from selected features              │
│     - Grow tree to maximum depth (no pruning)               │
│                           ↓                                 │
│  3. Repeat steps 1-2 for n_estimators trees                 │
│                           ↓                                 │
│  4. For prediction, aggregate outputs from all trees        │
│     (average for regression)                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Hyperparameters

| Parameter        | Type    | Default | Description                                 |
|------------------|---------|---------|---------------------------------------------|
| `n_estimators`   | int     | 100     | Number of trees in the forest                |
| `max_depth`      | int     | None    | Maximum depth of each tree (None = unlimited) |
| `min_samples_split` | int/float | 2   | Minimum samples required to split a node    |
| `min_samples_leaf` | int/float | 1    | Minimum samples required at a leaf node     |
| `max_features`   | int/float/str | 1.0 | Number of features to consider for best split |
| `bootstrap`      | bool    | True    | Whether to use bootstrap samples             |
| `max_samples`    | int/float | None  | Number of samples to draw for each tree     |
| `random_state`   | int     | None    | Seed for reproducibility                     |
| `n_jobs`         | int     | None    | Number of CPU cores to use (-1 for all)     |
| `oob_score`      | bool    | False   | Use out-of-bag samples to estimate R² score |

---

## Assumptions

- **Minimal assumptions** compared to parametric models
- No specific distribution assumptions for features or target
- Assumes that averaging diverse models improves prediction
- Works best when trees are uncorrelated (achieved through randomness)

---

## Evaluation Metrics

| Metric   | Formula                                   | Interpretation                        |
|----------|--------------------------------------------|---------------------------------------|
| **MAE**  | $\frac{1}{n}\sum |y_i - \hat{y}_i|$         | Average absolute prediction error     |
| **MSE**  | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$        | Average squared error                 |
| **RMSE** | $\sqrt{MSE}$                               | Root mean squared error               |
| **R²**   | $1 - \frac{SS_{res}}{SS_{tot}}$            | Proportion of variance explained      |
| **OOB Score** | Out-of-bag R² score                     | Cross-validation estimate without test set |

---

## Pros & Cons

| ✅ Advantages                | ❌ Disadvantages                       |
|-----------------------------|----------------------------------------|
| Handles non-linear relationships | Less interpretable (black box)      |
| Robust to outliers           | Slower prediction than single trees    |
| No feature scaling required  | Larger memory footprint                |
| Provides feature importance  | Can overfit on noisy datasets          |
| Reduces overfitting via bagging | Not suitable for extrapolation      |
| Handles missing values well  | Biased toward dominant classes (classification) |
| Works with mixed data types  | Computationally intensive for training |

---

## Implementation Example

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model initialization
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
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

# Out-of-bag score (if oob_score=True)
print(f"OOB Score: {model.oob_score_:.4f}")

# Feature importance
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
print("\nTop 5 Important Features:")
for idx in sorted_idx[:5]:
    print(f"Feature {idx}: {feature_importance[idx]:.4f}")
```

---

<div align="center">

**📚 Related:** [Decision Tree Regressor](../DecisionTree/Regressor/) | [Linear Regression](../Linear-Regression/) | [Polynomial Regression](../Polynomial-Regression/)

</div>
