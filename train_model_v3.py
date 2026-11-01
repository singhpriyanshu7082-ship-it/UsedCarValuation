import re
import os
import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/cleaned_car_data.csv"
MODEL_PATH = "models/car_price_model_v3.cbm"

df = pd.read_csv(DATA_PATH)

def get_brand(name):
    name = str(name).strip()
    return name.split()[0].lower()

df["brand"] = df["name"].apply(get_brand)
df["car_age"] = 2026 - df["year"]

features = [
    "name", "brand", "year", "car_age", "km_driven",
    "fuel", "seller_type", "transmission", "owner"
]

X = df[features].copy()
y = df["selling_price"].copy()

categorical_features = [
    "name", "brand", "fuel", "seller_type", "transmission", "owner"
]

for col in categorical_features:
    X[col] = X[col].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = CatBoostRegressor(
    iterations=1200,
    depth=7,
    learning_rate=0.04,
    loss_function="RMSE",
    eval_metric="R2",
    l2_leaf_reg=5,
    random_seed=42,
    verbose=100,
    allow_writing_files=False
)

model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
    eval_set=(X_test, y_test),
    early_stopping_rounds=150,
    verbose=100
)

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n========== CATBOOST MODEL RESULTS ==========")
print(f"R2 Score: {r2:.4f} ({r2 * 100:.2f}%)")
print(f"MAE: Rs. {mae:,.2f}")
print(f"RMSE: Rs. {rmse:,.2f}")

os.makedirs("models", exist_ok=True)
model.save_model(MODEL_PATH)

print("\nNew model saved at:", MODEL_PATH)
