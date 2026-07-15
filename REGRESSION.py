# Experiment: Linear Regression vs Ridge Regression vs Lasso Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# ------------------------------------------------------------
# 2. Split Dataset
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# 3. Feature Scaling
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 4. Create Models
# ------------------------------------------------------------
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.1, max_iter=10000)

# Display Lasso Configuration
print("=" * 60)
print("Lasso Model Configuration")
print("=" * 60)
print(lasso_model)

# ------------------------------------------------------------
# 5. Train Models
# ------------------------------------------------------------
linear_model.fit(X_train_scaled, y_train)
ridge_model.fit(X_train_scaled, y_train)
lasso_model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
# 6. Make Predictions
# ------------------------------------------------------------
linear_pred = linear_model.predict(X_test_scaled)
ridge_pred = ridge_model.predict(X_test_scaled)
lasso_pred = lasso_model.predict(X_test_scaled)

# ------------------------------------------------------------
# 7. Evaluation Function
# ------------------------------------------------------------
def evaluate_model(name, model, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    result = {
        "Model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    }

    if isinstance(model, Lasso):
        result["Alpha"] = model.alpha
        result["Max Iter"] = model.max_iter
    else:
        result["Alpha"] = "-"
        result["Max Iter"] = "-"

    return result

# ------------------------------------------------------------
# 8. Model Performance
# ------------------------------------------------------------
results = [
    evaluate_model("Linear Regression", linear_model, y_test, linear_pred),
    evaluate_model("Ridge Regression", ridge_model, y_test, ridge_pred),
    evaluate_model("Lasso Regression", lasso_model, y_test, lasso_pred)
]

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("Model Performance Comparison")
print("=" * 60)
print(results_df.to_string(index=False))


# ------------------------------------------------------------
# 9. Coefficient Comparison
# ------------------------------------------------------------
coef_df = pd.DataFrame({
    "Feature": diabetes.feature_names,
    "Linear Regression": np.round(linear_model.coef_, 2),
    "Ridge Regression": np.round(ridge_model.coef_, 2),
    "Lasso Regression": np.round(lasso_model.coef_, 2)
})

print("\n" + "=" * 60)
print("Coefficient Comparison")
print("=" * 60)
print(coef_df.to_string(index=False))

# ------------------------------------------------------------
# 10. Actual vs Predicted Scatter Plot
# ------------------------------------------------------------
plt.figure(figsize=(8,5))

plt.scatter(y_test, linear_pred,
            alpha=0.6,
            marker='o',
            label="Linear Regression")

plt.scatter(y_test, ridge_pred,
            alpha=0.6,
            marker='^',
            label="Ridge Regression")

plt.scatter(y_test, lasso_pred,
            alpha=0.6,
            marker='s',
            label="Lasso Regression")

ideal_line = np.linspace(min(y_test), max(y_test), 100)
plt.plot(ideal_line,
         ideal_line,
         color='black',
         linestyle='--',
         label='Perfect Fit')

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 11. R2 Score Bar Graph (Without Values on Bars)
# ------------------------------------------------------------
plt.figure(figsize=(7,4))

plt.bar(
    results_df["Model"],
    results_df["R2 Score"],
    color=['#3498db', '#2ecc71', '#e74c3c'],
    width=0.4
)

plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.title("R2 Score Comparison")
plt.ylim(0, 0.6)
plt.grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

