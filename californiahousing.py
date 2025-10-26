import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = fetch_california_housing(as_frame = True)
df = data.frame.copy()

df = df.rename(columns={"MedHouseVal" : "target"})

#Analize data
print("Shape: ", df.shape)
print("\nHead:")
print(df.head(10))

print("\nInfo: ")
df.info()

print("\nUseful information: ")
print(df.describe().T)


X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42)
print(X_train.shape, X_test.shape)

#Characteristics of the XGBoost model
xgb = XGBRegressor(
n_estimators=200,learning_rate=0.05,max_depth=6,subsample=0.8,
colsample_bytree = 0.8, random_state=42,n_jobs=-1,tree_method="hist",eval_metric="rmse")

eval_set = [(X_train, y_train), (X_test,y_test)]

xgb.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=True #show the progress
)

y_pred = xgb.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print(f"MSE:{mse:.4f}")

results = xgb.evals_result()

plt.figure()
plt.plot(results["validation_0"]["rmse"], label="train RMSE")
plt.plot(results["validation_1"]["rmse"], label="valid RMSE")
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("XGB Learning Curve")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)

min_v = min(y_test.min(), y_pred.min())
max_v = max(y_test.max(), y_pred.max())
plt.plot([min_v, max_v], [min_v, max_v])
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.show()


residuals = y_test - y_pred
plt.figure()
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (y_true - y_pred)")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.show()

from xgboost import plot_importance
plt.figure()
plot_importance(xgb, importance_type="gain", show_values=False)
plt.title("Feature Importance (gain)")
plt.tight_layout()
plt.show()
