# DecisionTreeRegressor

A **Decision Tree Regressor** is a supervised machine learning model used to predict **continuous values**.  
It works by splitting the data into branches based on feature thresholds, forming a tree-like structure of decisions. Each leaf node predicts the **average value** of the target for that branch.

---

## Type
- Supervised Learning  
- Regression  

---

## When to Use
- When the relationship between features and target is **non-linear**  
- When **feature interactions** are important  
- When you want a model that is **easy to interpret**  

---

## Input / Output
- **Input (X):** Numerical features (all categorical features must be encoded first)  
- **Output (y):** Continuous value  

---

## Preprocessing
- **Feature Scaling:** ❌ Not required  
- **Missing Values:** ❌ Must be handled (drop or impute)  
- **Categorical Data:** ⚠️ Encode using OneHotEncoder or LabelEncoder  
- **Outliers:** ✅ Decision Trees are mostly robust, but extreme outliers can affect splits  

---

## How It Works
1. Starts with all data at the root  
2. Tries all possible splits on all features  
3. Chooses the split that **reduces variance** the most (MSE by default)  
4. Recursively splits the data until a stopping condition is reached (max depth, min samples)  
5. Each leaf predicts the **mean target value** of the data points it contains  

---

## Important Parameters
| Parameter | Purpose |
|---------|---------|
| `max_depth` | Maximum depth of the tree (controls overfitting) |
| `min_samples_split` | Minimum number of samples required to split a node |
| `min_samples_leaf` | Minimum number of samples required at a leaf node |
| `max_features` | Number of features to consider at each split |
| `criterion` | Function to measure split quality (`squared_error` for regression) |

---

## Overfitting Control
- Limit `max_depth`  
- Increase `min_samples_leaf`  
- Increase `min_samples_split`  

---

## Evaluation Metrics
- **Mean Absolute Error (MAE)** – average magnitude of errors  
- **Mean Squared Error (MSE) / Root MSE** – penalizes larger errors  
- **R² Score** – percentage of variance explained  

---

## Advantages
- Works well for non-linear relationships  
- No need for feature scaling  
- Easy to interpret and visualize  

---

## Disadvantages
- Can **overfit** easily if not controlled  
- Sensitive to small variations in data  
- Poor extrapolation outside training range  
- Not ideal for very large datasets  

---

## Basic Example

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
