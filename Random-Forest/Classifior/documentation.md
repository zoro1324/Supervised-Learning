# Random Forest Classifier

> A supervised machine learning algorithm for predicting categorical labels using an ensemble of decision trees that aggregates predictions through majority voting to improve accuracy and reduce overfitting.

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
| **Library** | `sklearn.ensemble.RandomForestClassifier`     |

Random Forest Classifier is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes (majority vote) predicted by individual trees. It uses bootstrap aggregating (bagging) and random feature selection to create diverse trees, reducing variance and preventing overfitting while maintaining low bias.

---

## Use Cases

| Scenario                        | Suitability      |
|----------------------------------|------------------|
| Non-linear decision boundaries   | ✅ Excellent    |
| High-dimensional datasets         | ✅ Excellent    |
| Feature importance analysis       | ✅ Excellent    |
| Handling outliers                 | ✅ Excellent    |
| Imbalanced classes                | ✅ Excellent (with class_weight) |
| Mixed data types (numerical + categorical) | ✅ Excellent    |
| Multi-class classification        | ✅ Excellent    |
| Large-scale datasets              | ✅ Good         |
| Real-time predictions             | ⚠️ Moderate (slower than single trees) |
| Model interpretability            | ⚠️ Limited (black box) |

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
| Missing Value Handling| ⚠️ Optional | Can handle NaN values in some implementations             |
| Categorical Encoding  | ✅ Yes   | Use `LabelEncoder` or `OrdinalEncoder` for ordinal, `OneHotEncoder` for nominal |
| Outlier Treatment     | ❌ No    | Robust to outliers due to ensemble averaging               |
| Class Balancing       | ⚠️ Optional | Use `class_weight='balanced'` for imbalanced datasets      |

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
│     (majority voting for classification)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Hyperparameters

| Parameter        | Type    | Default | Description                                 |
|------------------|---------|---------|---------------------------------------------|
| `n_estimators`   | int     | 100     | Number of trees in the forest                |
| `criterion`      | str     | 'gini'  | Function to measure split quality ('gini' or 'entropy') |
| `max_depth`      | int     | None    | Maximum depth of each tree (None = unlimited) |
| `min_samples_split` | int/float | 2   | Minimum samples required to split a node    |
| `min_samples_leaf` | int/float | 1    | Minimum samples required at a leaf node     |
| `max_features`   | int/float/str | 'sqrt' | Number of features to consider for best split |
| `bootstrap`      | bool    | True    | Whether to use bootstrap samples             |
| `max_samples`    | int/float | None  | Number of samples to draw for each tree     |
| `class_weight`   | dict/str | None   | Weights for classes ('balanced' or custom dict) |
| `random_state`   | int     | None    | Seed for reproducibility                     |
| `n_jobs`         | int     | None    | Number of CPU cores to use (-1 for all)     |
| `oob_score`      | bool    | False   | Use out-of-bag samples to estimate accuracy |

---

## Assumptions

- **Minimal assumptions** compared to parametric models
- No specific distribution assumptions for features or target
- Assumes that averaging diverse models improves prediction
- Works best when trees are uncorrelated (achieved through randomness)
- No assumption of class separability (can handle complex boundaries)

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
| **Confusion Matrix** | Matrix of TP, TN, FP, FN            | Detailed error breakdown              |
| **OOB Score** | Out-of-bag accuracy score                  | Cross-validation estimate without test set |

**Legend:** TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative

---

## Pros & Cons

| ✅ Advantages                | ❌ Disadvantages                       |
|-----------------------------|----------------------------------------|
| Handles non-linear decision boundaries | Less interpretable (black box)      |
| Robust to outliers           | Slower prediction than single trees    |
| No feature scaling required  | Larger memory footprint                |
| Provides feature importance  | Can overfit on noisy datasets          |
| Reduces overfitting via bagging | Biased toward majority class without tuning |
| Handles missing values well  | Not suitable for extrapolation         |
| Works with mixed data types  | Computationally intensive for training |
| Excellent for imbalanced data (with class_weight) | Poor performance on very small datasets |
| Built-in cross-validation (OOB) | May struggle with highly imbalanced data |

---

## Implementation Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score)
import numpy as np

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model initialization
model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',  # Handle imbalanced classes
    oob_score=True,
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
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")

# For binary classification
if len(np.unique(y)) == 2:
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")

# Out-of-bag score (if oob_score=True)
print(f"OOB Score: {model.oob_score_:.4f}")

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

### Handling Imbalanced Classes

```python
# Method 1: Use class_weight parameter
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Automatically adjusts weights
    random_state=42
)

# Method 2: Custom class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

model = RandomForestClassifier(
    n_estimators=100,
    class_weight=class_weight_dict,
    random_state=42
)

# Method 3: Adjust decision threshold (for binary classification)
from sklearn.metrics import precision_recall_curve

# Get predicted probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

# Predict with custom threshold
y_pred_custom = (y_scores >= optimal_threshold).astype(int)
```

---

<div align="center">

**📚 Related:** [Decision Tree Classifier](../../DecisionTree/Classifier/) | [Random Forest Regressor](../Regressor/) | [Logistic Regression](#)

</div>
