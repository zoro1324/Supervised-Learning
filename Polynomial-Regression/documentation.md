# Polynomial Regression

> A supervised machine learning algorithm for predicting continuous numerical values by modeling a non-linear relationship between input features and the target variable using polynomial terms.

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
| **Library** | `sklearn.preprocessing.PolynomialFeatures`,<br>`sklearn.linear_model.LinearRegression` |

Polynomial Regression extends Linear Regression by adding polynomial terms to the input features, allowing the model to fit non-linear relationships between features (X) and the target variable (y). The model predicts target values by minimizing the error between actual and predicted values, typically using the Mean Squared Error (MSE) criterion.

---

## Use Cases

| Scenario                        | Suitability      |
|----------------------------------|------------------|
| Non-linear feature-target relationship | ✅ Excellent    |
| Curve fitting                    | ✅ Excellent    |
| Baseline for non-linear regression| ✅ Good         |
| Large-scale datasets              | ✅ Good         |
| High-degree polynomials           | ⚠️ Risk of overfitting |

---

## Input & Output

| Component      | Description                                      |
|--------------- |--------------------------------------------------|
| **Input (X)**  | Numerical feature matrix *(categorical features must be encoded)* |
| **Output (y)** | Continuous numerical values                      |

---

## Data Preprocessing

| Preprocessing Step     | Required | Notes                                                      |
|-----------------------|----------|------------------------------------------------------------|
| Feature Scaling       | ⚠️ Optional | Recommended for high-degree polynomials or Gradient Descent |
| Missing Value Handling| ✅ Yes   | Impute or remove missing values                            |
| Categorical Encoding  | ✅ Yes   | Use `OneHotEncoder` or similar                             |
| Outlier Treatment     | ⚠️ Optional | Improves performance, especially for sensitive models      |

---

## Algorithm Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Transform input features to polynomial features          │
│                           ↓                                 │
│  2. Fit a linear model to the transformed features           │
│                           ↓                                 │
│  3. Minimize error (e.g., MSE) to learn optimal parameters  │
│                           ↓                                 │
│  4. Predict target values for new data                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Hyperparameters

| Parameter        | Type    | Default | Description                                 |
|------------------|---------|---------|---------------------------------------------|
| `degree`         | int     | 2       | Degree of the polynomial features            |
| `include_bias`   | bool    | True    | Whether to include a bias (intercept) term   |
| `interaction_only`| bool   | False   | Only interaction features, no powers         |
| `fit_intercept`  | bool    | True    | Whether to calculate the intercept for the model |
| `n_jobs`         | int     | None    | Number of CPU cores to use                  |

---

## Assumptions

- The relationship between X and y can be approximated by a polynomial function
- No or little multicollinearity
- Homoscedasticity (constant variance of errors)
- Errors are normally distributed

---

## Evaluation Metrics

| Metric   | Formula                                   | Interpretation                        |
|----------|--------------------------------------------|---------------------------------------|
| **MAE**  | $\frac{1}{n}\sum |y_i - \hat{y}_i|$         | Average absolute prediction error     |
| **MSE**  | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$        | Average squared error                 |
| **RMSE** | $\sqrt{MSE}$                               | Root mean squared error               |
| **R²**   | $1 - \frac{SS_{res}}{SS_{tot}}$            | Proportion of variance explained      |

---

## Pros & Cons

| ✅ Advantages                | ❌ Disadvantages                       |
|-----------------------------|----------------------------------------|
| Captures non-linear patterns | Risk of overfitting with high degree   |
| Flexible curve fitting       | Less interpretable than linear models  |
| Simple to implement          | Sensitive to outliers                  |
| Works with linear algorithms | May require feature scaling            |

---

## Implementation Example

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Polynomial feature transformation
poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Model initialization
model = LinearRegression()

# Training
model.fit(X_train_poly, y_train)

# Prediction
y_pred = model.predict(X_test_poly)

# Evaluation
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE:  {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
```

---

<div align="center">

**📚 Related:** [Linear Regression](../Linear-Regression/) | [Decision Tree Regressor](../DecisionTree/Regressor/) | [Ridge Regression](#)

</div>
