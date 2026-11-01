import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load cleaned dataset
df = pd.read_csv("data/cleaned_car_data.csv")

# Create folder for graphs
Path("plots").mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# Graph 1: Price distribution
plt.figure(figsize=(9, 5))
upper_limit = df["selling_price"].quantile(0.99)

sns.histplot(
    data=df[df["selling_price"] <= upper_limit],
    x="selling_price",
    bins=30,
    kde=True
)

plt.title("Distribution of Used Car Prices")
plt.xlabel("Selling Price (INR)")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.savefig("plots/price_distribution.png")
plt.show()
plt.close()

# Graph 2: Median price by fuel type
plt.figure(figsize=(9, 5))

fuel_prices = (
    df.groupby("fuel")["selling_price"]
    .median()
    .sort_values(ascending=False)
)

sns.barplot(x=fuel_prices.index, y=fuel_prices.values)

plt.title("Median Car Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Median Selling Price (INR)")
plt.tight_layout()
plt.savefig("plots/price_by_fuel.png")
plt.show()
plt.close()

# Graph 3: Price versus kilometres driven
plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="km_driven",
    y="selling_price",
    hue="transmission",
    alpha=0.6
)

plt.title("Car Price vs Kilometres Driven")
plt.xlabel("Kilometres Driven")
plt.ylabel("Selling Price (INR)")
plt.tight_layout()
plt.savefig("plots/price_vs_km.png")
plt.show()
plt.close()

print("All 3 graphs created successfully!")
print("Check the plots folder.")
