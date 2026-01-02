# Machine Learning Workflow Guide

A comprehensive step-by-step guide for building, training, and evaluating machine learning models from problem definition to deployment.

---

## Step 1: Problem Understanding & Definition

**Objective:** Define the business problem and ML task type

### Key Decisions:

**What is your target variable?**
- Numerical value → **Regression**
- Category/Label → **Classification**
- Pattern discovery → **Clustering**

**Problem Type Selection Guide:**

| Task Type | Prediction Type | Use Case | Example |
|-----------|-----------------|----------|---------|
| **Regression** | Continuous numerical | Predict amounts, rates, prices | Salary prediction, house price estimation |
| **Classification** | Discrete categories | Categorize into groups | Email spam detection, disease diagnosis |
| **Clustering** | Group similarities | Discover patterns without labels | Customer segmentation, document grouping |

**Additional Considerations:**
- Problem scope and business impact
- Available resources and timeline
- Data availability and quality expectations
- Accuracy vs. interpretability requirements

---

## Step 2: Data Collection & Loading

**Objective:** Gather and prepare data for analysis

### Data Sources:
- CSV/Excel files
- Relational databases (SQL)
- APIs and web services
- Data warehouses

### Implementation:
```python
import pandas as pd

# Load data
data = pd.read_csv('dataset.csv')

# Explore data
print(data.shape)              # (rows, columns)
print(data.info())             # Data types, missing values
print(data.describe())         # Statistical summary
print(data.head())             # First few rows
```

### What to Check:
- Data shape (rows × columns)
- Data types (numeric, object, datetime)
- Missing value percentage
- Target variable distribution
- Feature ranges and scales

---

## Step 3: Data Preprocessing

**Objective:** Clean and prepare data for modeling

### 3.1 Handle Missing Values

**Decision Guide:**

| Missing % | Strategy | When to Use |
|-----------|----------|------------|
| < 5% | Remove rows | Minimal data loss, sufficient samples |
| 5-20% | Imputation | Moderate missing data |
| > 20% | Drop feature | Feature unreliable/uninformative |

**Imputation Methods:**
- **Mean/Median:** Continuous numerical features
- **Mode:** Categorical features
- **Forward Fill/Backward Fill:** Time-series data
- **Predictive models:** Complex missing patterns

### 3.2 Encode Categorical Data

**Encoding Strategy Selection:**

| Encoding Type | Use Case | When to Choose |
|---------------|----------|----------------|
| **Label Encoding** | Tree-based models | 2-3 categories; tree-based algorithms |
| **One-Hot Encoding** | Linear/Distance models | Few categories (< 10); linear models, KNN, SVM |
| **Ordinal Encoding** | Ordered categories | Data has natural ordering (Low→Medium→High) |
| **Target Encoding** | High-cardinality | Many categories (> 10); at risk of overfitting |

### 3.3 Feature Scaling

**When to Scale:**

| Model Type | Scaling Required? | Recommended Method |
|------------|-------------------|-------------------|
| Decision Trees, Random Forest | ❌ No | Not needed (tree-based) |
| Linear Regression, Ridge, Lasso | ✅ Yes | StandardScaler |
| KNN, SVM, Neural Networks | ✅ Yes | StandardScaler or MinMaxScaler |
| Logistic Regression | ✅ Yes | StandardScaler |
| Gradient Boosting (XGBoost) | ❌ No | Not required |

**Scaling Methods:**

| Method | Formula | Best For | Range |
|--------|---------|----------|-------|
| **StandardScaler** | $(x - \mu) / \sigma$ | Most algorithms | [-∞, +∞] |
| **MinMaxScaler** | $(x - x_{min}) / (x_{max} - x_{min})$ | Neural networks | [0, 1] |
| **RobustScaler** | $(x - median) / IQR$ | Datasets with outliers | [-∞, +∞] |
| **MaxAbsScaler** | $x / max(\|x\|)$ | Sparse data | [-1, 1] |

---

## Step 4: Feature Selection & Engineering

**Objective:** Use relevant features and create meaningful new ones

### Feature Selection Methods:

| Method | Description | When to Use |
|--------|-------------|------------|
| **Correlation Analysis** | Remove highly correlated features | Quick initial screening |
| **SelectKBest** | Keep top K features by statistical score | Automated feature ranking |
| **Recursive Feature Elimination (RFE)** | Iteratively remove least important features | When computational budget allows |
| **Model-Based Importance** | Use trained model's feature importance | After initial model training |

### Feature Engineering:
- Create polynomial features for non-linear relationships
- Interaction terms (multiply related features)
- Domain-specific transformations (log, square root)
- Time-based features (day of week, month, season)
- Aggregate features from grouped data

**Best Practice:** Engineer features based on domain knowledge, not just statistical significance.

---

## Step 5: Train-Test Split

**Objective:** Create separate datasets for training and evaluation

### Split Strategy Selection:

| Data Size | Train/Test Split | Validation Method |
|-----------|------------------|-------------------|
| < 10,000 samples | 80/20 or 70/30 | Simple split + validation set |
| 10,000-100,000 | 80/20 | Cross-validation (5-fold) |
| > 100,000 | 80/20 or 90/10 | Time-series: sequential split |
| **Time-Series** | Sequential split | Do NOT shuffle; respect temporal order |

### Implementation:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 20% test, 80% train
    random_state=42,         # Reproducibility
    stratify=y               # For classification: balance classes
)
```

---

## Step 6: Model Selection

**Objective:** Choose appropriate algorithm(s) based on problem type

### For Regression (Predicting Continuous Values):

| Model | Best For | Complexity | Training Speed | Interpretability |
|-------|----------|-----------|-----------------|-----------------|
| **Linear Regression** | Simple linear relationships | Low | Very Fast | ✅ Excellent |
| **Ridge/Lasso Regression** | Many features, prevent overfitting | Low | Very Fast | ✅ Good |
| **Polynomial Regression** | Non-linear relationships | Medium | Fast | ✅ Good |
| **Decision Tree Regressor** | Non-linear, feature interactions | Medium | Fast | ✅ Good |
| **Random Forest Regressor** | Complex patterns, robust | High | Moderate | ⚠️ Moderate |
| **Gradient Boosting** | Best accuracy, complex data | Very High | Slow | ❌ Poor |

**Regression Model Selection Guide:**

Start with **Linear Regression**:
- If R² < 0.7 and data appears non-linear → Try **Random Forest** or **Polynomial Regression**
- If feature importance needed → Use **Random Forest**
- If many features & overfitting → Use **Ridge/Lasso**
- If maximum accuracy needed → Try **Gradient Boosting**

### For Classification (Predicting Categories):

| Model | Best For | Training Speed | Multiclass Support |
|-------|----------|-----------------|-------------------|
| **Logistic Regression** | Linear decision boundaries | Very Fast | ✅ Yes |
| **Decision Tree Classifier** | Non-linear, interpretable | Fast | ✅ Yes |
| **Random Forest Classifier** | Complex patterns, robust | Moderate | ✅ Yes |
| **Support Vector Machine (SVM)** | High-dimensional data | Slow | ✅ Yes (one-vs-rest) |
| **K-Nearest Neighbors (KNN)** | Small datasets, local patterns | Slow (prediction) | ✅ Yes |
| **Gradient Boosting** | Maximum accuracy, imbalanced data | Slow | ✅ Yes |

---

## Step 7: Model Training

**Objective:** Fit the model to training data

### Implementation:
```python
# Train the model
model.fit(X_train, y_train)

# The model learns optimal parameters by minimizing error
# on the training dataset
```

### Key Points:
- Use **training data only** to avoid data leakage
- Monitor training time and computational resources
- For large datasets, consider sampling strategies
- Implement early stopping for boosting algorithms

---

## Step 8: Model Evaluation

**Objective:** Assess model performance on unseen data

### Regression Metrics:

| Metric | Formula | Interpretation | When to Use |
|--------|---------|-----------------|------------|
| **MAE** (Mean Absolute Error) | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Average absolute error | Interpretable, robust to outliers |
| **MSE** (Mean Squared Error) | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Average squared error | Penalizes large errors heavily |
| **RMSE** (Root Mean Squared Error) | $\sqrt{MSE}$ | Error in original units | Most commonly used |
| **R² Score** | $1 - \frac{SS_{res}}{SS_{tot}}$ | % variance explained (0-1) | Overall model fit quality |
| **MAPE** (Mean Absolute % Error) | $\frac{1}{n}\sum\|\frac{y_i - \hat{y}_i}{y_i}\|$ | Percentage error | Compare across different scales |

**Regression Performance Interpretation:**
- R² > 0.8 → Excellent
- R² 0.6-0.8 → Good
- R² 0.4-0.6 → Moderate (improvement needed)
- R² < 0.4 → Poor (reconsider approach)

### Classification Metrics:

| Metric | When to Use | Interpretation |
|--------|------------|-----------------|
| **Accuracy** | Balanced classes | % correct predictions |
| **Precision** | False positives costly | Of predicted positives, how many correct? |
| **Recall** | False negatives costly | Of actual positives, how many found? |
| **F1 Score** | Balanced metric needed | Harmonic mean of precision & recall |
| **AUC-ROC** | Binary classification | Model's ability to distinguish classes |
| **Confusion Matrix** | Understand error types | True/False Positives/Negatives |

### Evaluation Implementation:
```python
# Always evaluate on TEST SET (unseen data)
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
```

---

## Step 9: Hyperparameter Tuning

**Objective:** Optimize model performance by adjusting hyperparameters

### Tuning Strategy:

| Approach | Use When | Pros | Cons |
|----------|----------|------|------|
| **GridSearchCV** | Limited hyperparameters | Exhaustive search | Slow for large spaces |
| **RandomizedSearchCV** | Many hyperparameters | Faster than grid search | May miss optimal combination |
| **Bayesian Optimization** | Complex search spaces | Efficient, finds good parameters | More complex to implement |
| **Manual Tuning** | Understanding desired behavior | Fast, interpretable | Requires domain expertise |

### Implementation:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    model, 
    param_grid, 
    cv=5,              # 5-fold cross-validation
    scoring='r2'
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
```

### Common Hyperparameters by Model:

**Random Forest/Decision Tree:**
- `max_depth`: Limit tree depth to prevent overfitting
- `min_samples_split`: Minimum samples for splitting nodes
- `n_estimators`: Number of trees (forest only)

**Linear Models (Ridge/Lasso):**
- `alpha`: Regularization strength (higher = more regularization)

**SVM:**
- `C`: Regularization parameter (inverse strength)
- `kernel`: 'linear', 'rbf', 'poly'

**Neural Networks:**
- Learning rate, batch size, number of layers, units per layer

---

## Step 10: Final Model & Prediction

**Objective:** Save trained model and generate predictions on new data

### Implementation:
```python
import joblib

# Save the best model
joblib.dump(best_model, 'model.pkl')

# Load model later
loaded_model = joblib.load('model.pkl')

# Predict on new data
new_predictions = loaded_model.predict(new_data)

# Get prediction probabilities (classification)
probabilities = loaded_model.predict_proba(new_data)
```

### Deployment Considerations:
- Ensure consistent data preprocessing for new data
- Document feature engineering steps
- Version control your models
- Monitor model performance over time

---

## Step 11: Deployment (Optional)

**Objective:** Deploy model to production environment

### Deployment Options:

| Option | Best For | Complexity |
|--------|----------|-----------|
| **REST API** (Flask, FastAPI) | Web services, integration | Medium |
| **Web Application** | User-facing predictions | Medium-High |
| **Batch Predictions** | Large-scale processing | Low |
| **Cloud Platform** (AWS, GCP, Azure) | Scalable, managed services | High |
| **MLOps Pipeline** (Docker, Kubernetes) | Production-grade systems | Very High |

### Flask API Example:
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
```

---

## Critical Best Practices

| Practice | Impact | Implementation |
|----------|--------|-----------------|
| **Use test set only for final evaluation** | Prevents optimistic bias | Never tune hyperparameters on test set |
| **Validate model on unseen data** | Ensures generalization | Use cross-validation during development |
| **Document preprocessing steps** | Production consistency | Save scaler, encoder objects; apply to new data identically |
| **Handle class imbalance** | Prevent majority class bias | Use stratified split, class weights, or resampling |
| **Monitor data drift** | Catch model degradation | Compare new data distribution to training data |
| **Version your models and data** | Reproducibility | Use Git for code, track data versions |

---

## Model Selection Decision Tree

```
├─ Regression Problem
│  ├─ Linear relationship → Linear Regression
│  ├─ Non-linear, interpretable → Decision Tree / Polynomial
│  ├─ Non-linear, many features → Random Forest
│  └─ Maximum accuracy needed → Gradient Boosting
│
└─ Classification Problem
   ├─ Linearly separable → Logistic Regression
   ├─ Non-linear, balanced → Decision Tree / Random Forest
   ├─ Imbalanced classes → Random Forest w/ class_weight
   ├─ High dimensions, small dataset → SVM
   └─ Interpretability critical → Logistic Regression / Decision Tree
```

---

## Summary Checklist

- [ ] Problem clearly defined (regression/classification/clustering)
- [ ] Data collected and loaded successfully
- [ ] Missing values handled appropriately
- [ ] Categorical data encoded
- [ ] Features scaled (if required by model)
- [ ] Train-test split performed (stratified if classification)
- [ ] Baseline model trained and evaluated
- [ ] Hyperparameters tuned via cross-validation
- [ ] Final model evaluated on test set
- [ ] Model saved for future predictions
- [ ] Results documented and reproducible
- [ ] Deployment plan established (if applicable)

---

**For more information, see individual algorithm documentation in respective folders.**
