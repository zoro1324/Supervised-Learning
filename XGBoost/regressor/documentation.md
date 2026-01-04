# XGBoost Regressor

> A supervised machine learning algorithm for predicting continuous numerical values using an ensemble of gradient-boosted decision trees that builds models sequentially to minimize prediction errors through advanced regularization and optimization techniques.

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
| **Library** | `xgboost.XGBRegressor`                        |

XGBoost (eXtreme Gradient Boosting) Regressor is an advanced implementation of gradient boosting that builds an ensemble of decision trees sequentially. Each new tree attempts to correct the errors made by the previous trees by minimizing a differentiable loss function using gradient descent. XGBoost includes regularization terms, handles missing values automatically, supports parallel processing, and implements tree pruning using a depth-first approach with a max_depth parameter, making it one of the most powerful and efficient machine learning algorithms for regression tasks.

---

## Use Cases

| Scenario                        | Suitability      |
|----------------------------------|------------------|
| Non-linear relationships         | ✅ Excellent    |
| High-dimensional datasets         | ✅ Excellent    |
| Feature importance analysis       | ✅ Excellent    |
| Handling missing values           | ✅ Excellent    |
| Structured/tabular data           | ✅ Excellent    |
| Mixed data types (numerical + categorical) | ✅ Excellent    |
| Large-scale datasets              | ✅ Excellent    |
| Kaggle competitions               | ✅ Excellent    |
| Time-series forecasting           | ✅ Good         |
| Real-time predictions             | ✅ Good         |
| Model interpretability            | ⚠️ Moderate (with SHAP values) |
| Very small datasets               | ⚠️ Limited (prone to overfitting) |
| Image or text data                | ⚠️ Limited (deep learning preferred) |

---

## Input & Output

| Component      | Description                                      |
|--------------- |--------------------------------------------------|
| **Input (X)**  | Numerical and categorical feature matrix *(categorical features should be encoded)* |
| **Output (y)** | Continuous numerical values (real numbers) |

---

## Data Preprocessing

| Preprocessing Step     | Required | Notes                                                      |
|-----------------------|----------|------------------------------------------------------------|
| Feature Scaling       | ❌ No    | Tree-based models are invariant to feature scaling         |
| Missing Value Handling| ❌ No    | XGBoost handles NaN values automatically                   |
| Categorical Encoding  | ✅ Yes   | Use `LabelEncoder` or `OrdinalEncoder` for tree-based learning, or set `enable_categorical=True` (XGBoost 1.6+) |
| Outlier Treatment     | ⚠️ Optional | Robust to outliers but extreme values may affect performance |
| Target Transformation | ⚠️ Optional | Log transformation for skewed targets can improve performance |

---

## Algorithm Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Initialize predictions with a constant value            │
│     (typically mean of target for regression)               │
│                           ↓                                 │
│  2. For iteration t = 1 to n_estimators:                    │
│     a. Calculate residuals (gradients) for current model    │
│     b. Fit a new decision tree to predict residuals         │
│     c. Add tree to ensemble with learning rate (eta)        │
│     d. Update predictions: F_t = F_(t-1) + η × h_t(x)       │
│                           ↓                                 │
│  3. Apply regularization (L1/L2) to prevent overfitting     │
│     - Penalize tree complexity (gamma, alpha, lambda)       │
│     - Limit tree depth and leaf weights                     │
│                           ↓                                 │
│  4. Final prediction = F_0 + Σ(η × h_t(x))                  │
│     (weighted sum of all trees)                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Concepts:**
- **Gradient Boosting:** Each tree corrects errors (residuals) from previous trees
- **Regularization:** Prevents overfitting through tree complexity penalties
- **Learning Rate (eta):** Controls contribution of each tree (smaller = more robust)
- **Early Stopping:** Monitors validation set to prevent overfitting

---

## Hyperparameters

### Core Tree Parameters

| Parameter        | Type    | Default | Description                                 |
|------------------|---------|---------|---------------------------------------------|
| `n_estimators`   | int     | 100     | Number of boosting rounds (trees to build)  |
| `learning_rate` (`eta`) | float | 0.3 | Step size shrinkage (range: 0-1, typical: 0.01-0.3) |
| `max_depth`      | int     | 6       | Maximum depth of each tree                  |
| `min_child_weight` | float | 1       | Minimum sum of instance weight in a child   |
| `subsample`      | float   | 1.0     | Fraction of samples used for each tree      |
| `colsample_bytree` | float | 1.0     | Fraction of features used for each tree     |
| `colsample_bylevel` | float | 1.0   | Fraction of features used at each level     |
| `colsample_bynode` | float | 1.0    | Fraction of features used at each node      |

### Regularization Parameters

| Parameter        | Type    | Default | Description                                 |
|------------------|---------|---------|---------------------------------------------|
| `gamma` (`min_split_loss`) | float | 0 | Minimum loss reduction required to split   |
| `alpha` (`reg_alpha`) | float | 0    | L1 regularization term on weights           |
| `lambda` (`reg_lambda`) | float | 1  | L2 regularization term on weights           |
| `max_delta_step` | float   | 0       | Maximum delta step for weight estimation    |

### Learning Task Parameters

| Parameter        | Type    | Default | Description                                 |
|------------------|---------|---------|---------------------------------------------|
| `objective`      | str     | 'reg:squarederror' | Loss function ('reg:squarederror', 'reg:squaredlogerror', 'reg:pseudohubererror', 'reg:absoluteerror') |
| `eval_metric`    | str     | None    | Evaluation metric ('rmse', 'mae', 'mape', 'rmsle') |
| `seed` (`random_state`) | int | 0     | Random seed for reproducibility             |

### System & Other Parameters

| Parameter        | Type    | Default | Description                                 |
|------------------|---------|---------|---------------------------------------------|
| `n_jobs`         | int     | 1       | Number of parallel threads (-1 for all)     |
| `tree_method`    | str     | 'auto'  | Tree construction algorithm ('auto', 'exact', 'approx', 'hist', 'gpu_hist') |
| `early_stopping_rounds` | int | None | Stop if no improvement for N rounds        |
| `verbosity`      | int     | 1       | Verbosity of messages (0=silent, 3=debug)   |
| `enable_categorical` | bool | False  | Enable categorical feature support (XGBoost 1.6+) |

---

## Assumptions

- **Minimal parametric assumptions** for data distribution
- Assumes that sequential error correction improves predictions
- Target variable should be continuous (for regression)
- Works best with structured/tabular data
- Assumes features are informative for the target variable
- No assumption of linear relationships between features and target
- Benefits from features having meaningful patterns (not pure noise)

---

## Evaluation Metrics

| Metric   | Formula                                   | Interpretation                        |
|----------|--------------------------------------------|---------------------------------------|
| **MSE (Mean Squared Error)** | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | Average squared difference (penalizes large errors) |
| **RMSE (Root Mean Squared Error)** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | Square root of MSE (same units as target) |
| **MAE (Mean Absolute Error)** | $\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | Average absolute difference (robust to outliers) |
| **R² Score (Coefficient of Determination)** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Proportion of variance explained (1=perfect, 0=baseline) |
| **Adjusted R²** | $1 - \frac{(1-R^2)(n-1)}{n-p-1}$ | R² adjusted for number of features |
| **MAPE (Mean Absolute Percentage Error)** | $\frac{100\%}{n}\sum_{i=1}^{n}\|\frac{y_i - \hat{y}_i}{y_i}\|$ | Average percentage error |
| **RMSLE (Root Mean Squared Log Error)** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\log(1+y_i) - \log(1+\hat{y}_i))^2}$ | RMSE on log scale (for skewed targets) |

**Legend:** $y_i$ = actual value, $\hat{y}_i$ = predicted value, $\bar{y}$ = mean of actual values, $n$ = number of samples, $p$ = number of features

---

## Pros & Cons

| ✅ Advantages                | ❌ Disadvantages                       |
|-----------------------------|----------------------------------------|
| State-of-the-art performance | Requires careful hyperparameter tuning |
| Handles missing values automatically | Can overfit on small/noisy datasets |
| Built-in regularization      | Less interpretable than linear models  |
| Feature importance provided  | Longer training time than simple models |
| No feature scaling required  | Memory intensive for large datasets    |
| Supports parallel processing | Sensitive to outliers in target variable |
| Early stopping prevents overfitting | Not suitable for unstructured data (images/text) |
| Handles mixed data types     | May not extrapolate well beyond training range |
| Robust to multicollinearity  | Requires more computational resources  |
| Excellent for tabular data   | Black box model (though SHAP helps)    |
| Custom objective functions   | Learning curve can be steep for beginners |

---

## Implementation Example

### Basic Implementation

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                             r2_score, mean_absolute_percentage_error)
import numpy as np

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model initialization
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE:   {mse:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"MAE:   {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Calculate MAPE (avoid division by zero)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE:  {mape:.4f}")

# Feature importance
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
print("\nTop 5 Important Features:")
for idx in sorted_idx[:5]:
    print(f"Feature {idx}: {feature_importance[idx]:.4f}")
```

### Advanced Implementation with Early Stopping and Cross-Validation

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# Create train/validation/test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Model with early stopping
model = xgb.XGBRegressor(
    n_estimators=1000,  # Set high, early stopping will determine actual number
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='reg:squarederror',
    eval_metric='rmse',
    random_state=42,
    n_jobs=-1
)

# Train with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score:.4f}")

# Prediction
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
print(f"\nTest Set Performance:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# Cross-validation
cv_scores = cross_val_score(
    model, X_train, y_train, 
    cv=5, 
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
print(f"\nCross-validation RMSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

### Hyperparameter Tuning with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# Initialize model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

# Grid search
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:")
print(grid_search.best_params_)
print(f"\nBest cross-validation score: {-grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

### Using DMatrix for Better Performance

```python
import xgboost as xgb

# Create DMatrix (XGBoost's internal data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'alpha': 0.1,
    'lambda': 1.0,
    'seed': 42
}

# Train with early stopping
evals = [(dtrain, 'train'), (dval, 'validation')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=50
)

# Prediction
y_pred = model.predict(dtest)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\nTest RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")
```

### Feature Importance Visualization

```python
import matplotlib.pyplot as plt
import xgboost as xgb

# Train model
model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Plot feature importance (different types)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Weight: number of times a feature is used in trees
xgb.plot_importance(model, importance_type='weight', ax=axes[0], max_num_features=10)
axes[0].set_title('Feature Importance (Weight)')

# Gain: average gain when the feature is used for splitting
xgb.plot_importance(model, importance_type='gain', ax=axes[1], max_num_features=10)
axes[1].set_title('Feature Importance (Gain)')

# Cover: average coverage when the feature is used for splitting
xgb.plot_importance(model, importance_type='cover', ax=axes[2], max_num_features=10)
axes[2].set_title('Feature Importance (Cover)')

plt.tight_layout()
plt.show()
```

### Handling Skewed Target Variable

```python
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Log transformation for skewed target
y_train_log = np.log1p(y_train)  # log(1 + y) to handle zeros
y_val_log = np.log1p(y_val)

# Train on log-transformed target
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    objective='reg:squarederror',  # or use 'reg:squaredlogerror'
    random_state=42
)

model.fit(
    X_train, y_train_log,
    eval_set=[(X_val, y_val_log)],
    early_stopping_rounds=50,
    verbose=False
)

# Predict and inverse transform
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # exp(y) - 1

# Evaluate on original scale
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE (original scale): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
```

### Model Interpretation with SHAP

```python
import shap
import xgboost as xgb

# Train model
model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot (feature importance)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Force plot for single prediction
shap.force_plot(
    explainer.expected_value, 
    shap_values[0], 
    X_test.iloc[0],
    feature_names=feature_names
)

# Dependence plot (relationship between feature and prediction)
shap.dependence_plot(
    "feature_name", 
    shap_values, 
    X_test, 
    feature_names=feature_names
)
```

---

<div align="center">

**📚 Related:** [XGBoost Classifier](#) | [Random Forest Regressor](../../Random-Forest/Regressor/) | [Gradient Boosting Regressor](#)

</div>
