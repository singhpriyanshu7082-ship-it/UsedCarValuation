# UsedCar AI - Smart Used Car Valuation System



## Project Overview

UsedCar AI is a Machine Learning based web application that predicts the estimated resale price of used cars based on their specifications and historical car data.



The application is developed using Python, CatBoost and Streamlit.



## Project Objectives

\- Predict the estimated selling price of used cars.

\- Analyze historical used car data.

\- Compare different Machine Learning algorithms.

\- Display interactive data visualizations.



## Dataset Information

\- Dataset: CarDekho Used Car Dataset

\- Original Records: 4,340

\- Cleaned Records: 3,577

\- Original Features: 8

\- Target Variable: selling\_price



## Machine Learning Model

The project uses CatBoost Regressor for price prediction.



### Feature Engineering

Two additional features are created:

\- brand

\- car\_age



The final model uses 9 input features.



### Model Performance



| Metric | Result |

|---|---:|

| RÂ² Score | 78.16% |

| MAE | â‚¹1,28,691.01 |

| RMSE | â‚¹2,65,250.22 |



## Algorithms Compared

\- Random Forest Regressor

\- Extra Trees Regressor

\- CatBoost Regressor



## Technologies Used

\- Python

\- Pandas

\- NumPy

\- Matplotlib

\- Seaborn

\- Scikit-learn

\- CatBoost

\- Streamlit

\- Joblib



## Application Features

1\. Home Page

2\. Price Prediction

3\. Analytics Dashboard

4\. Model Performance

5\. About Project



## Machine Learning Workflow

1\. Dataset Collection

2\. Data Cleaning and Preprocessing

3\. Exploratory Data Analysis

4\. Feature Engineering

5\. Model Training

6\. Model Evaluation

7\. Model Saving

8\. Streamlit Web Application Development

9\. Price Prediction



## How to Run the Project



Install the required libraries:



```bash

pip install streamlit pandas numpy matplotlib seaborn scikit-learn catboost joblib

```



Run the application:



```bash

streamlit run app.py

```



Open the local URL displayed in your terminal, usually:



http://localhost:8501



## Future Improvements

\- Improve prediction accuracy.

\- Add more vehicle-related features.

\- Deploy the application online.

\- Improve user interface and experience.



## Disclaimer

Predicted prices are estimates based on historical dataset patterns. Actual resale prices may vary depending on vehicle condition, location and market demand.



## Author

Developed as a College Machine Learning Project.



