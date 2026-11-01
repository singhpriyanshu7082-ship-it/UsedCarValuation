
import pandas as pd
import numpy as np
from pathlib import Path

# Load dataset
files = [
    f for f in Path("data").glob("*.csv")
    if f.name.lower() != "cleaned_car_data.csv"
]

if not files:
    raise FileNotFoundError("Original CSV data folder mein nahi mili.")

df = pd.read_csv(files[0])

print("=" * 40)
print("USED CAR DATA ANALYSIS")
print("=" * 40)

print("\n1. Dataset shape:")
print(df.shape)

print("\n2. Data types:")
print(df.dtypes)

print("\n3. Missing values:")
print(df.isnull().sum())

print("\n4. Duplicate rows:", df.duplicated().sum())

print("\n5. Selling price statistics:")
print(df["selling_price"].describe())

prices = df["selling_price"].dropna().to_numpy()

print("\n6. NumPy analysis:")
print("Average price:", np.mean(prices))
print("Median price:", np.median(prices))
print("Minimum price:", np.min(prices))
print("Maximum price:", np.max(prices))

# Clean dataset
df.columns = df.columns.str.strip()
df = df.drop_duplicates().copy()

for col in ["year", "selling_price", "km_driven"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["year", "selling_price", "km_driven"]
)

df = df[
    (df["selling_price"] > 0)
    & (df["km_driven"] >= 0)
    & (df["year"] >= 1900)
    & (df["year"] <= 2026)
].copy()

df.to_csv("data/cleaned_car_data.csv", index=False)

print("\nCleaning completed!")
print("Cleaned dataset shape:", df.shape)
print("Saved: data/cleaned_car_data.csv")
