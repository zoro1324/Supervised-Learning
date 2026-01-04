# XGBoost Classifier

> A supervised machine learning algorithm for predicting categorical labels using an ensemble of gradient-boosted decision trees that builds models sequentially to minimize classification errors through advanced regularization and optimization techniques.

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
| **Task**    | Classification                                |
| **Library** | `xgboost.XGBClassifier`                       |

XGBoost (eXtreme Gradient Boosting) Classifier is an advanced implementation of gradient boosting that builds an ensemble of decision trees sequentially. Each new tree attempts to correct the classification errors made by the previous trees by minimizing a differentiable loss function (log loss for binary, softmax for multi-class) using gradient descent. XGBoost includes regularization terms, handles missing values automatically, supports parallel processing, and implements tree pruning using a depth-first approach with a max_depth parameter, making it one of the most powerful and efficient machine learning algorithms for classification tasks.

---

## Use Cases

| Scenario                        | Suitability      |
|----------------------------------|------------------|
| Non-linear decision boundaries   | ✅ Excellent    |
| High-dimensional datasets         | ✅ Excellent    |
| Feature importance analysis       | ✅ Excellent    |
| Handling missing values           | ✅ Excellent    |
| Imbalanced classes                | ✅ Excellent (with scale_pos_weight) |
| Structured/tabular data           | ✅ Excellent    |
| Mixed data types (numerical + categorical) | ✅ Excellent    |
| Multi-class classification        | ✅ Excellent    |
| Large-scale datasets              | ✅ Excellent    |
| Kaggle competitions               | ✅ Excellent    |
| Binary classification             | ✅ Excellent    |
| Real-time predictions             | ✅ Good         |
| Probability calibration           | ✅ Good         |
| Model interpretability            | ⚠️ Moderate (with SHAP values) |
| Very small datasets               | ⚠️ Limited (prone to overfitting) |
| Image or text data                | ⚠️ Limited (deep learning preferred) |

---

## Input & Output

| Component      | Description                                      |
|--------------- |--------------------------------------------------|
| **Input (X)**  | Numerical and categorical feature matrix *(categorical features should be encoded)* |
| **Output (y)** | Discrete categorical labels (binary or multi-class) |

---

## Data Preprocessing

| Preprocessing Step     | Required | Notes                                                      |
|-----------------------|----------|------------------------------------------------------------|
| Feature Scaling       | ❌ No    | Tree-based models are invariant to feature scaling         |
| Missing Value Handling| ❌ No    | XGBoost handles NaN values automatically                   |
| Categorical Encoding  | ✅ Yes   | Use `LabelEncoder` or `OrdinalEncoder` for tree-based learning, or set `enable_categorical=True` (XGBoost 1.6+) |
| Outlier Treatment     | ⚠️ Optional | Robust to outliers but extreme values may affect performance |
| Class Balancing       | ⚠️ Optional | Use `scale_pos_weight` for binary or `class_weight` for imbalanced datasets |

---

## Algorithm Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Initialize predictions with a constant value            │
│     (log-odds for binary, class probs for multi-class)      │
│                           ↓                                 │
│  2. For iteration t = 1 to n_estimators:                    │
│     a. Calculate gradients and hessians for loss function   │
│     b. Fit a new decision tree to predict pseudo-residuals  │
│     c. Add tree to ensemble with learning rate (eta)        │
│     d. Update predictions: F_t = F_(t-1) + η × h_t(x)       │
│                           ↓                                 │
│  3. Apply regularization (L1/L2) to prevent overfitting     │
│     - Penalize tree complexity (gamma, alpha, lambda)       │
│     - Limit tree depth and leaf weights                     │
│                           ↓                                 │
│  4. Final prediction:                                       │
│     - Binary: Apply sigmoid to get probabilities            │
│     - Multi-class: Apply softmax to get class probabilities │
└─────────────────────────────────────────────────────────────┘
```

**Key Concepts:**
- **Gradient Boosting:** Each tree corrects errors from previous trees
- **Regularization:** Prevents overfitting through tree complexity penalties
- **Learning Rate (eta):** Controls contribution of each tree (smaller = more robust)
- **Early Stopping:** Monitors validation set to prevent overfitting
- **Scale Pos Weight:** Balances positive and negative classes in binary classification

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
| `objective`      | str     | 'binary:logistic' | Loss function ('binary:logistic', 'multi:softmax', 'multi:softprob') |
| `eval_metric`    | str     | None    | Evaluation metric ('logloss', 'error', 'auc', 'aucpr', 'merror', 'mlogloss') |
| `scale_pos_weight` | float | 1       | Balance of positive and negative weights (for imbalanced data) |
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
- Target variable should be categorical (discrete classes)
- Works best with structured/tabular data
- Assumes features are informative for the target variable
- No assumption of linear relationships or class separability
- Benefits from features having meaningful patterns (not pure noise)

---

## Evaluation Metrics

| Metric   | Formula                                   | Interpretation                        |
|----------|--------------------------------------------|---------------------------------------|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$         | Overall correct predictions           |
| **Precision** | $\frac{TP}{TP + FP}$                       | Correct positive predictions          |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$          | True positives found                  |
| **F1 Score** | $2 \times \frac{Precision \times Recall}{Precision + Recall}$ | Harmonic mean of precision & recall |
| **Specificity** | $\frac{TN}{TN + FP}$                    | True negatives found                  |
| **AUC-ROC** | Area Under ROC Curve                       | Model's discriminative ability        |
| **AUC-PR** | Area Under Precision-Recall Curve          | Performance on imbalanced data        |
| **Log Loss (Cross-Entropy)** | $-\frac{1}{n}\sum_{i=1}^{n}[y_i\log(p_i) + (1-y_i)\log(1-p_i)]$ | Penalizes confident wrong predictions |
| **Confusion Matrix** | Matrix of TP, TN, FP, FN            | Detailed error breakdown              |
| **Multi-class Log Loss** | $-\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{M}y_{ij}\log(p_{ij})$ | Log loss for multi-class problems |

**Legend:** TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative

---

## Pros & Cons

| ✅ Advantages                | ❌ Disadvantages                       |
|-----------------------------|----------------------------------------|
| State-of-the-art performance | Requires careful hyperparameter tuning |
| Handles missing values automatically | Can overfit on small/noisy datasets |
| Built-in regularization      | Less interpretable than linear models  |
| Feature importance provided  | Longer training time than simple models |
| No feature scaling required  | Memory intensive for large datasets    |
| Supports parallel processing | Sensitive to outliers in features      |
| Early stopping prevents overfitting | Not suitable for unstructured data (images/text) |
| Handles mixed data types     | May require extensive tuning for optimal results |
| Excellent for imbalanced data (with scale_pos_weight) | Requires more computational resources  |
| Handles multicollinearity    | Black box model (though SHAP helps)    |
| Probability calibration available | Learning curve can be steep for beginners |
| Works for binary and multi-class | May not generalize well with limited data |

---

## Implementation Example

### Basic Implementation (Binary Classification)

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, log_loss)
import numpy as np

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model initialization
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluation
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")
print(f"Log Loss:  {log_loss(y_test, y_pred_proba):.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
print("\nTop 5 Important Features:")
for idx in sorted_idx[:5]:
    print(f"Feature {idx}: {feature_importance[idx]:.4f}")
```

### Advanced Implementation with Early Stopping

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

# Create train/validation/test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Model with early stopping
model = xgb.XGBClassifier(
    n_estimators=1000,  # Set high, early stopping will determine actual number
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='binary:logistic',
    eval_metric='logloss',
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
y_pred_proba = model.predict_proba(X_test)

# Evaluation
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
print(f"\nTest Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:  {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")
```

### Handling Imbalanced Classes

```python
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import numpy as np

# Method 1: Use scale_pos_weight parameter (for binary classification)
# Calculate ratio of negative to positive samples
neg_samples = np.sum(y_train == 0)
pos_samples = np.sum(y_train == 1)
scale_pos_weight = neg_samples / pos_samples

model = xgb.XGBClassifier(
    n_estimators=100,
    scale_pos_weight=scale_pos_weight,  # Balance classes
    objective='binary:logistic',
    random_state=42
)

model.fit(X_train, y_train)

# Method 2: Use sample weights
sample_weights = compute_sample_weight('balanced', y_train)

model = xgb.XGBClassifier(
    n_estimators=100,
    objective='binary:logistic',
    random_state=42
)

model.fit(X_train, y_train, sample_weight=sample_weights)

# Method 3: Adjust decision threshold
from sklearn.metrics import precision_recall_curve

# Get predicted probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.4f}")

# Predict with custom threshold
y_pred_custom = (y_scores >= optimal_threshold).astype(int)

# Evaluate
from sklearn.metrics import classification_report
print("\nClassification Report (Custom Threshold):")
print(classification_report(y_test, y_pred_custom))
```

### Multi-class Classification

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score)
import numpy as np

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model for multi-class
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective='multi:softprob',  # Returns probabilities for each class
    num_class=len(np.unique(y)),  # Number of classes
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Multi-class AUC-ROC (one-vs-rest)
auc_ovr = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
print(f"AUC-ROC (OvR): {auc_ovr:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
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
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1
)

# Grid search
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',  # Can use 'accuracy', 'roc_auc', 'f1', etc.
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:")
print(grid_search.best_params_)
print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
from sklearn.metrics import accuracy_score
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Using DMatrix for Better Performance

```python
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

# Create DMatrix (XGBoost's internal data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
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
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test AUC-ROC:  {auc_roc:.4f}")
```

### Feature Importance Visualization

```python
import matplotlib.pyplot as plt
import xgboost as xgb

# Train model
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
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

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import numpy as np

# Initialize model
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1
)

# Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation with multiple metrics
cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
cv_roc_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"Cross-validation Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
print(f"Cross-validation F1 Score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
print(f"Cross-validation AUC-ROC:  {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std():.4f})")
```

### Model Interpretation with SHAP

```python
import shap
import xgboost as xgb

# Train model
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values is for the positive class
# For multi-class, shap_values is a list of arrays (one per class)

# Summary plot (feature importance)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Force plot for single prediction (binary classification)
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

# Waterfall plot for individual prediction
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=feature_names
    )
)
```

### Probability Calibration

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import xgboost as xgb

# Train base XGBoost model
base_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

base_model.fit(X_train, y_train)

# Calibrate probabilities using Platt scaling or isotonic regression
calibrated_model = CalibratedClassifierCV(
    base_model, 
    method='sigmoid',  # or 'isotonic'
    cv=5
)

calibrated_model.fit(X_train, y_train)

# Get predictions
y_pred_proba_uncal = base_model.predict_proba(X_test)[:, 1]
y_pred_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]

# Evaluate calibration using Brier score (lower is better)
brier_uncal = brier_score_loss(y_test, y_pred_proba_uncal)
brier_cal = brier_score_loss(y_test, y_pred_proba_cal)

print(f"Brier Score (Uncalibrated): {brier_uncal:.4f}")
print(f"Brier Score (Calibrated):   {brier_cal:.4f}")

# Plot calibration curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, y_pred_proba_uncal, n_bins=10)
prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_pred_proba_cal, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
plt.plot(prob_pred_uncal, prob_true_uncal, 's-', label='Uncalibrated')
plt.plot(prob_pred_cal, prob_true_cal, 's-', label='Calibrated')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True)
plt.show()
```

---

<div align="center">

**📚 Related:** [XGBoost Regressor](../regressor/) | [Random Forest Classifier](../../Random-Forest/Classifior/) | [Gradient Boosting Classifier](#)

</div>
