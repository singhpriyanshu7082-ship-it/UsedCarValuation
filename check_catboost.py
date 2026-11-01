import pandas as pd
import catboost

df = pd.read_csv("data/cleaned_car_data.csv")

print("Dataset shape:", df.shape)
print("CatBoost version:", catboost.__version__)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())
