import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Loading dataset...")

df = pd.read_csv("data/cleaned_car_data.csv")
df = df.dropna(subset=["selling_price"])

X = df.drop(columns=["selling_price"])
y = df["selling_price"]

# Identify categorical and numerical features
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

print("Features:", X.columns.tolist())
print("Categorical features:", categorical_cols)
print("Numerical features:", numerical_cols)

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), categorical_cols),
    ("num", "passthrough", numerical_cols)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Two algorithms
algorithms = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )
}

best_model = None
best_mae = float("inf")
best_name = ""

for name, algorithm in algorithms.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", algorithm)
    ])

    model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n", name, "Results")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))
    print("R2 Percentage:", round(r2 * 100, 2), "%")

    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_name = name

# Save best model
Path("models").mkdir(exist_ok=True)

joblib.dump(best_model, "models/car_price_model.pkl")

print("\n================================")
print("BEST MODEL:", best_name)
print("BEST MAE:", round(best_mae, 2))
print("Best model saved successfully!")
print("================================")
