# Linear Regression

> A supervised machine learning algorithm for predicting continuous numerical values by fitting a linear relationship between input features and the target variable.

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
| **Library** | `sklearn.linear_model.LinearRegression`        |

Linear Regression models the relationship between one or more input features (X) and a continuous target variable (y) by fitting a straight line. The model predicts target values by minimizing the error between actual and predicted values, typically using the Mean Squared Error (MSE) criterion.

---

## Use Cases

| Scenario                        | Suitability      |
|----------------------------------|------------------|
| Linear feature-target relationship | ✅ Excellent    |
| Trend analysis                    | ✅ Excellent    |
| Baseline regression model         | ✅ Good         |
| Large-scale datasets              | ✅ Good         |
| Non-linear relationships          | ⚠️ Consider non-linear models |

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
| Feature Scaling       | ⚠️ Optional | Recommended for Gradient Descent; not needed for Normal Equation |
| Missing Value Handling| ✅ Yes   | Impute or remove missing values                            |
| Categorical Encoding  | ✅ Yes   | Use `OneHotEncoder` or similar                             |
| Outlier Treatment     | ⚠️ Optional | Improves performance, especially for sensitive models      |

---

## Algorithm Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Initialize model parameters (weights and bias)          │
│                           ↓                                 │
│  2. Fit a straight line: y = mX + c                         │
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
| `fit_intercept`  | bool    | True    | Whether to calculate the intercept for the model |
| `normalize`      | bool    | False   | Normalize input features (deprecated)       |
| `n_jobs`         | int     | None    | Number of CPU cores to use                  |

---

## Assumptions

- Linear relationship between X and y
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
| Simple and fast             | Performs poorly with non-linear data   |
| Easy to interpret           | Sensitive to outliers                  |
| Works well for linear data  | Assumes linear relationship            |
| Baseline for regression     | May underfit complex relationships     |

---

## Implementation Example

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

# Model initialization
model = LinearRegression()

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

**📚 Related:** [Decision Tree Regressor](../DecisionTree/Regressor/) | [Ridge Regression](#) | [Lasso Regression](#)

</div>
