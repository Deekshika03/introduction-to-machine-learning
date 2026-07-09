#Experiment: Linear Regression vs Ridge Regression vs Lasso Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 2. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create models
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.1, max_iter=10000)

# 5. Train models
linear_model.fit(X_train_scaled, y_train)
ridge_model.fit(X_train_scaled, y_train)
lasso_model.fit(X_train_scaled, y_train)

# 6. Make predictions
linear_pred = linear_model.predict(X_test_scaled)
ridge_pred = ridge_model.predict(X_test_scaled)
lasso_pred = lasso_model.predict(X_test_scaled)

# 7. Evaluation function
def evaluate_model(name, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        "Model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    }

# 8. Store results
results = [
    evaluate_model("Linear Regression", y_test, linear_pred),
    evaluate_model("Ridge Regression", y_test, ridge_pred),
    evaluate_model("Lasso Regression", y_test, lasso_pred)
]
results_df = pd.DataFrame(results)

print("=" * 60)
print("Model Performance Comparison:")
print("=" * 60)
print(results_df.to_string(index=False))

# 9. Compare coefficients

coef_df = pd.DataFrame({
    "Feature": diabetes.feature_names,
    "Linear Regression": np.round(linear_model.coef_, 2),
    "Ridge Regression": np.round(ridge_model.coef_, 2),
    "Lasso Regression": np.round(lasso_model.coef_, 2)
})

print("\n" + "=" * 60)
print("Coefficient Comparison:")
print("=" * 60)
print(coef_df.to_string(index=False))

# 10. Plot actual vs predicted values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, linear_pred, alpha=0.6, label="Linear Regression", marker='o')
plt.scatter(y_test, ridge_pred, alpha=0.6, label="Ridge Regression", marker='^')
plt.scatter(y_test, lasso_pred, alpha=0.6, label="Lasso Regression", marker='s')

# Add an ideal reference line
ideal_line = np.linspace(min(y_test), max(y_test), 100)
plt.plot(ideal_line, ideal_line, color='black', linestyle='--', label='Perfect Fit')

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

 
# 11. Bar graph for R2 score
plt.figure(figsize=(7, 4))
bars = plt.bar(results_df["Model"], results_df["R2 Score"], color=['#3498db', '#2ecc71', '#e74c3c'], width=0.4)
plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.title("R2 Score Comparison")
plt.ylim(0, 0.6) # Gives breathing room for values around 0.45

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=0) # Kept at 0 since names fit perfectly without tilt
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
