import pandas as pd
import joblib

model = joblib.load("models/car_price_model.pkl")

print("\n===== USED CAR PRICE PREDICTOR =====")

name = input("Enter car name: ")
year = int(input("Manufacturing year: "))
km = int(input("Kilometres driven: "))
fuel = input("Fuel type (Petrol/Diesel/CNG/LPG/Electric): ")
seller = input("Seller type (Individual/Dealer/Trustmark Dealer): ")
transmission = input("Transmission (Manual/Automatic): ")
owner = input("Owner type (First Owner/Second Owner/Third Owner/Fourth & Above Owner/Test Drive Car): ")

car = pd.DataFrame([{
    "name": name,
    "year": year,
    "km_driven": km,
    "fuel": fuel,
    "seller_type": seller,
    "transmission": transmission,
    "owner": owner
}])

price = model.predict(car)[0]

print("\n================================")
print("Car:", name)
print("Estimated Selling Price: ₹", round(price, 2))
print("================================")
