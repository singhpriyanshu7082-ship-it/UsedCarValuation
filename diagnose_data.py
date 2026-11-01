import pandas as pd
import numpy as np

print("=" * 55)
print("USED CAR DATASET - DIAGNOSTIC REPORT")
print("=" * 55)

df = pd.read_csv("data/cleaned_car_data.csv")

print("\n1. DATASET INFORMATION")
print("Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

print("\n2. MISSING VALUES")
print(df.isnull().sum())

print("\n3. TARGET PRICE STATISTICS")
print(df["selling_price"].describe())

print("\n4. PRICE PERCENTILES")
print(df["selling_price"].quantile(
    [0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1]
))

print("\n5. EXTREME PRICE VALUES")
print("Lowest 10 prices:")
print(df.nsmallest(10, "selling_price")[
    ["name", "year", "selling_price"]
])

print("\nHighest 10 prices:")
print(df.nlargest(10, "selling_price")[
    ["name", "year", "selling_price"]
])

print("\n6. CAR NAME ANALYSIS")
print("Unique car names:", df["name"].nunique())
print("\nMost common car names:")
print(df["name"].value_counts().head(15))

print("\n7. CATEGORICAL FEATURE DISTRIBUTION")

for col in ["fuel", "seller_type", "transmission", "owner"]:
    print(f"\n{col}:")
    print(df[col].value_counts())

print("\n8. MANUFACTURING YEAR ANALYSIS")
print(df["year"].value_counts().sort_index())

print("\n9. DUPLICATE ROWS")
print("Duplicate rows:", df.duplicated().sum())

print("\n10. NUMERICAL CORRELATION")
print(df.select_dtypes(include=np.number).corr())

print("\n" + "=" * 55)
print("DIAGNOSTIC COMPLETED SUCCESSFULLY")
print("=" * 55)
