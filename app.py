import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from catboost import CatBoostRegressor

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="UsedCar AI | Smart Car Valuation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "car_price_model_v3.cbm"
DATA_PATH = BASE_DIR / "data" / "cleaned_car_data.csv"

# -----------------------------------------
# CUSTOM CSS
# -----------------------------------------

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 30px;
    border-radius: 16px;
    background: linear-gradient(120deg, #14213d, #1d3557);
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 38px;
}
.hero p {
    color: #d5e2f0;
    font-size: 16px;
}
div[data-testid="stMetric"] {
    background: #1c2333;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #30394b;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------
# LOAD MODEL AND DATASET
# -----------------------------------------

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    return model

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# -----------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------

st.sidebar.title("🚗 UsedCar AI")
st.sidebar.caption("Smart Used Car Valuation System")

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "Price Prediction",
        "Analytics Dashboard",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.info("Machine Learning Based Car Price Prediction")


# -----------------------------------------
# LOAD RESOURCES
# -----------------------------------------

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Application loading error: {e}")
    st.stop()


# -----------------------------------------
# HOME PAGE
# -----------------------------------------

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>🚗 UsedCar AI</h1>
        <p>Smart Used Car Valuation System</p>
        <p>
        Predict the estimated resale value of your car
        using Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Welcome to UsedCar AI")

    st.write("""
    UsedCar AI is a Machine Learning based web application
    that estimates the resale price of used cars based on
    their specifications and historical car market data.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Records", f"{len(df):,}")

    with col2:
        st.metric(
            "Average Car Price",
            f"₹{df['selling_price'].mean():,.0f}"
        )

    with col3:
        st.metric("ML Algorithm", "CatBoost Regressor")

    st.divider()

    st.subheader("What Can You Do?")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 💰 Price Prediction")
        st.write("Enter car details and get an estimated resale price.")

        st.markdown("### 📊 Analytics Dashboard")
        st.write("Explore car prices, fuel types and other dataset insights.")

    with c2:
        st.markdown("### 🧠 Model Performance")
        st.write("Understand the performance of our trained ML algorithms.")

        st.markdown("### ℹ️ About Project")
        st.write("Learn about the project, methodology and technologies.")

    st.divider()
    st.caption("UsedCar AI | Smart Car Valuation System")


# -----------------------------------------
# PRICE PREDICTION PAGE
# -----------------------------------------

elif page == "Price Prediction":

    st.title("💰 Used Car Price Prediction")
    st.write("Enter your car details to estimate its resale value.")

    st.divider()

    with st.form("prediction_form"):

        st.subheader("Enter Your Car Details")

        col1, col2 = st.columns(2)

        with col1:

            name = st.text_input(
                "Car Name",
                value="Maruti Swift Dzire VDI"
            )

            year = st.number_input(
                "Manufacturing Year",
                min_value=1992,
                max_value=2020,
                value=2018,
                step=1
            )

            km = st.number_input(
                "Kilometers Driven",
                min_value=0,
                max_value=1000000,
                value=60000,
                step=1000
            )

            fuel = st.selectbox(
                "Fuel Type",
                ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
            )

        with col2:

            seller = st.selectbox(
                "Seller Type",
                ["Individual", "Dealer", "Trustmark Dealer"]
            )

            transmission = st.selectbox(
                "Transmission",
                ["Manual", "Automatic"]
            )

            owner = st.selectbox(
                "Owner Type",
                [
                    "First Owner",
                    "Second Owner",
                    "Third Owner",
                    "Fourth & Above Owner",
                    "Test Drive Car"
                ]
            )

        submitted = st.form_submit_button(
            "🚀 Predict Car Price",
            use_container_width=True
        )

    if submitted:

        if not name.strip():
            st.warning("Please enter a valid car name.")

        else:

            clean_name = name.strip()
            brand = clean_name.split()[0].lower()
            car_age = 2026 - year

            car = pd.DataFrame([{
                "name": clean_name,
                "brand": brand,
                "year": year,
                "car_age": car_age,
                "km_driven": km,
                "fuel": fuel,
                "seller_type": seller,
                "transmission": transmission,
                "owner": owner
            }])

            try:

                price = float(model.predict(car)[0])

                if price >= 0:

                    st.session_state["prediction_result"] = {
                        "price": price,
                        "name": name,
                        "year": year,
                        "km": km,
                        "fuel": fuel,
                        "seller": seller,
                        "transmission": transmission,
                        "owner": owner
                    }

                else:
                    st.warning("Model returned a negative price estimate.")

            except Exception as e:
                st.error(f"Prediction Error: {e}")

    if "prediction_result" in st.session_state:

        result = st.session_state["prediction_result"]

        st.divider()
        st.success("Prediction Completed Successfully!")

        st.metric(
            "Estimated Selling Price",
            f"₹ {result['price']:,.2f}"
        )

        st.subheader("Selected Car Details")

        details1, details2 = st.columns(2)

        with details1:
            st.write(f"**Car:** {result['name']}")
            st.write(f"**Year:** {result['year']}")
            st.write(f"**Kilometers:** {result['km']:,}")

        with details2:
            st.write(f"**Fuel:** {result['fuel']}")
            st.write(f"**Seller:** {result['seller']}")
            st.write(f"**Transmission:** {result['transmission']}")
            st.write(f"**Owner:** {result['owner']}")

        st.caption(
            "This is an estimated price, not a guaranteed market value."
        )

        if st.button("Clear Prediction"):
            del st.session_state["prediction_result"]
            st.rerun()


# -----------------------------------------
# ANALYTICS DASHBOARD
# -----------------------------------------

elif page == "Analytics Dashboard":

    st.title("📊 Car Analytics Dashboard")

    st.write("Explore insights from the cleaned used car dataset.")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Cars", f"{len(df):,}")

    with col2:
        st.metric(
            "Average Price",
            f"₹{df['selling_price'].mean():,.0f}"
        )

    with col3:
        st.metric(
            "Median Price",
            f"₹{df['selling_price'].median():,.0f}"
        )

    with col4:
        st.metric(
            "Maximum Price",
            f"₹{df['selling_price'].max():,.0f}"
        )

    st.divider()

    st.subheader("⛽ Fuel Type Distribution")

    fuel_counts = df["fuel"].value_counts()

    st.bar_chart(fuel_counts)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🚘 Transmission Distribution")

        transmission_counts = df["transmission"].value_counts()

        st.bar_chart(transmission_counts)

    with col2:

        st.subheader("🏢 Seller Type Distribution")

        seller_counts = df["seller_type"].value_counts()

        st.bar_chart(seller_counts)

    st.divider()

    st.subheader("💰 Car Price Distribution")

    price_distribution = pd.cut(
        df["selling_price"],
        bins=12
    ).value_counts().sort_index()

    price_distribution.index = price_distribution.index.astype(str)

    st.bar_chart(price_distribution)

    st.divider()

    st.subheader("📅 Year-wise Average Car Price")

    yearly_price = (
        df.groupby("year")["selling_price"]
        .mean()
        .sort_index()
    )

    st.line_chart(yearly_price)

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# -----------------------------------------
# MODEL PERFORMANCE PAGE
# -----------------------------------------

elif page == "Model Performance":

    st.title("🧠 Machine Learning Model Performance")

    st.write("Evaluation of the trained regression algorithms.")

    st.divider()

    st.subheader("Selected Model")

    st.success("CatBoost Regressor")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", "₹1,28,691.01")

    with col2:
        st.metric("RMSE", "₹2,65,250.22")

    with col3:
        st.metric("R² Score", "78.16%")

    st.divider()

    st.subheader("Algorithm Comparison")

    comparison = pd.DataFrame({
        "Algorithm": [
            "Random Forest",
            "Extra Trees",
            "CatBoost"
        ],
        "MAE": [
            164504.53,
            163208.21,
            128691.01
        ],
        "RMSE": [
            410504.57,
            410553.38,
            265250.22
        ],
        "R2 Score": [
            0.4769,
            0.4768,
            0.7816
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Understanding Model Metrics")

    st.markdown("""
    **1. MAE (Mean Absolute Error)**

    Average absolute difference between actual and predicted car prices.
    Lower MAE is better.

    **2. RMSE (Root Mean Squared Error)**

    Measures prediction errors while giving more importance to larger errors.
    Lower RMSE is better.

    **3. R² Score (Coefficient of Determination)**

    Measures how much variation in selling prices is explained by the model.
    A higher score is generally better.

    **Current Model R²: 0.7816**

    The model explains approximately 78.16% of the variation
    in the test dataset.
    """)

    st.info(
        "CatBoost was selected because it achieved better test-set performance."
    )


# -----------------------------------------
# ABOUT PROJECT PAGE
# -----------------------------------------

elif page == "About Project":

    st.title("ℹ️ About UsedCar AI")

    st.write("""
    UsedCar AI is a Machine Learning based project developed
    to estimate the resale price of used cars.
    """)

    st.divider()

    st.subheader("🎯 Project Objective")

    st.write("""
    The objective is to predict the estimated selling price
    of a used car using its specifications and historical data.
    """)

    st.subheader("📂 Dataset Information")

    st.write("""
    Dataset: CarDekho Used Car Dataset

    Cleaned records: 3,577

    Original features: 8

    Target variable: selling_price
    """)

    st.subheader("⚙️ Technologies Used")

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "NumPy",
            "Pandas",
            "Matplotlib",
            "Seaborn",
            "Scikit-learn",
            "Joblib",
            "Streamlit"
        ],
        "Purpose": [
            "Programming Language",
            "Numerical Operations",
            "Data Analysis",
            "Data Visualization",
            "Statistical Visualization",
            "Machine Learning",
            "Model Saving",
            "Web Application"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🔄 Machine Learning Workflow")

    st.markdown("""
    1. Dataset Collection
    2. Data Cleaning
    3. Exploratory Data Analysis
    4. Data Preprocessing
    5. Train-Test Split
    6. Model Training
    7. Model Evaluation
    8. Model Saving
    9. Web Application Development
    """)

    st.subheader("🚀 Future Improvements")

    st.markdown("""
    - Improve model prediction performance.
    - Add prediction history.
    - Generate downloadable reports.
    - Add CSV export.
    - Deploy the application online.
    """)

    st.divider()

    st.caption(
        "UsedCar AI | College Machine Learning Project"
    )
