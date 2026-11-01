
import pandas as pd
from pathlib import Path

files = list(Path("data").glob("*.csv"))

print("CSV files found:", files)

if files:
    df = pd.read_csv(files[0])

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset size:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())
else:
    print("Data folder mein CSV file nahi mili.")

