# Multinomial Naive Bayes

> A supervised machine learning algorithm for classification tasks based on Bayes' theorem, particularly effective for discrete features like word counts in text classification.

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
| **Library** | `sklearn.naive_bayes.MultinomialNB`           |

Multinomial Naive Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption of conditional independence between features. It is particularly suited for classification with discrete features (e.g., word counts, term frequencies). The algorithm calculates the probability of each class given the input features and predicts the class with the highest probability.

---

## Use Cases

| Scenario                        | Suitability      |
|----------------------------------|------------------|
| Text classification (spam detection) | ✅ Excellent    |
| Document categorization          | ✅ Excellent    |
| Sentiment analysis               | ✅ Excellent    |
| Topic classification             | ✅ Good         |
| Discrete/count-based features    | ✅ Excellent    |
| Small to medium datasets         | ✅ Good         |
| Real-time predictions            | ✅ Excellent    |
| Continuous features              | ⚠️ Use GaussianNB instead |

---

## Input & Output

| Component      | Description                                      |
|--------------- |--------------------------------------------------|
| **Input (X)**  | Discrete/count-based feature matrix *(typically non-negative integers)* |
| **Output (y)** | Categorical class labels                         |

---

## Data Preprocessing

| Preprocessing Step     | Required | Notes                                                      |
|-----------------------|----------|------------------------------------------------------------|
| Text Vectorization    | ✅ Yes   | Use `CountVectorizer` or `TfidfVectorizer` for text data   |
| Non-negative Features | ✅ Yes   | All features must be non-negative (counts/frequencies)     |
| Missing Value Handling| ✅ Yes   | Impute or remove missing values                            |
| Feature Scaling       | ❌ No    | Not required for Naive Bayes                               |
| Categorical Encoding  | ✅ Yes   | Encode categorical features if not already numeric         |

---

## Algorithm Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Calculate prior probabilities for each class             │
│                           ↓                                 │
│  2. Calculate conditional probabilities for each feature     │
│     given each class                                        │
│                           ↓                                 │
│  3. Apply Bayes' theorem to compute posterior probabilities │
│                           ↓                                 │
│  4. Predict class with highest posterior probability        │
└─────────────────────────────────────────────────────────────┘
```

**Mathematical Foundation:**

$$P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}$$

Where:
- $P(y|X)$ = Posterior probability of class y given features X
- $P(X|y)$ = Likelihood of features X given class y
- $P(y)$ = Prior probability of class y
- $P(X)$ = Evidence (constant for all classes)

---

## Hyperparameters

| Parameter        | Type    | Default | Description                                 |
|------------------|---------|---------|---------------------------------------------|
| `alpha`          | float   | 1.0     | Additive (Laplace/Lidstone) smoothing parameter |
| `fit_prior`      | bool    | True    | Whether to learn class prior probabilities   |
| `class_prior`    | array   | None    | Prior probabilities of classes (if not learned) |
| `force_alpha`    | bool    | True    | Whether to add alpha to feature log probabilities |

---

## Assumptions

- **Conditional Independence**: Features are conditionally independent given the class (naive assumption)
- **Feature Distribution**: Features follow a multinomial distribution
- **Non-negative Features**: All feature values must be non-negative (counts or frequencies)

---

## Evaluation Metrics

| Metric         | Formula                                   | Interpretation                        |
|----------------|-------------------------------------------|---------------------------------------|
| **Accuracy**   | $\frac{TP + TN}{TP + TN + FP + FN}$       | Overall correctness                   |
| **Precision**  | $\frac{TP}{TP + FP}$                      | Accuracy of positive predictions      |
| **Recall**     | $\frac{TP}{TP + FN}$                      | Coverage of actual positives          |
| **F1-Score**   | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ | Harmonic mean of precision & recall   |
| **ROC-AUC**    | Area under ROC curve                      | Trade-off between TPR and FPR         |

---

## Pros & Cons

| ✅ Advantages                | ❌ Disadvantages                       |
|-----------------------------|----------------------------------------|
| Fast training and prediction | Strong independence assumption         |
| Works well with small datasets | Poor estimator (probability calibration) |
| Handles high-dimensional data | Cannot learn feature interactions      |
| Performs well with text data | Requires non-negative features         |
| Scales well to large datasets | Sensitive to irrelevant features      |
| Probabilistic predictions    | Zero probability problem (mitigated by smoothing) |

---

## Implementation Example

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Example with text data
# Assuming 'texts' contains text data and 'labels' contains corresponding classes

# Text vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model initialization
model = MultinomialNB(alpha=1.0, fit_prior=True)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Probability predictions
y_pred_proba = model.predict_proba(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

---

<div align="center">

**📚 Related:** [Logistic Regression](../Logistic-Regression/) | [Decision Tree Classifier](../DecisionTree/Classifier/) | [Random Forest Classifier](../Random-Forest/Classifior/)

</div>
