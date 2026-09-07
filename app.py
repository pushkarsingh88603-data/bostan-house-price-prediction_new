import streamlit as st
import pandas as pd
import pickle
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"
COLUMNS_PATH = BASE_DIR / "columns.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(COLUMNS_PATH, "rb") as file:
        columns = pickle.load(file)

except FileNotFoundError:
    st.error(
        "❌ model.pkl or columns.pkl was not found.\n\n"
        "Make sure both files are inside the same folder as app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏠 House Price AI")

    st.markdown("---")

    st.subheader("📌 Project")

    st.write(
        "A Machine Learning application that predicts "
        "house prices using a trained regression model."
    )

    st.markdown("---")

    st.subheader("🧠 Machine Learning")

    st.write("✅ Data Cleaning")
    st.write("✅ Feature Engineering")
    st.write("✅ Feature Scaling")
    st.write("✅ Regression")
    st.write("✅ Model Evaluation")
    st.write("✅ Model Deployment")

    st.markdown("---")

    st.subheader("🛠️ Technologies")

    st.write("🐍 Python")
    st.write("📊 Pandas")
    st.write("🔢 NumPy")
    st.write("🤖 Scikit-learn")
    st.write("🚀 Streamlit")


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🏠 House Price Prediction")

st.caption(
    "Machine Learning powered house price estimation "
    "using a trained regression model."
)

st.markdown("---")


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.header("📊 Project Overview")

overview1, overview2, overview3, overview4 = st.columns(4)

with overview1:
    st.metric(
        label="🤖 Model Type",
        value="Regression"
    )

with overview2:
    st.metric(
        label="📥 Input Features",
        value=len(columns)
    )

with overview3:
    st.metric(
        label="⚙️ Framework",
        value="Streamlit"
    )

with overview4:
    st.metric(
        label="🐍 Language",
        value="Python"
    )


st.markdown("---")


# ============================================================
# PROPERTY INFORMATION
# ============================================================

st.header("🏡 Property Information")

st.write(
    "Enter the property characteristics below. "
    "The trained machine learning model will use these "
    "values to estimate the house price."
)


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = {
    "CRIM": "Crime Rate",
    "ZN": "Residential Land Zone (%)",
    "INDUS": "Non-Retail Business Area (%)",
    "CHAS": "Charles River Boundary",
    "NOX": "Nitric Oxide Concentration",
    "RM": "Average Number of Rooms",
    "AGE": "Age of Property (%)",
    "DIS": "Distance to Employment Centers",
    "RAD": "Highway Accessibility",
    "TAX": "Property Tax Rate",
    "PTRATIO": "Pupil-Teacher Ratio",
    "B": "Black Population Index",
    "LSTAT": "Lower Income Population (%)"
}


# ============================================================
# INPUT FORM
# ============================================================

data = {}

with st.form("house_prediction_form"):

    # Create 3 columns
    col1, col2, col3 = st.columns(3)

    for i, column in enumerate(columns):

        if column in feature_names:
            label = f"{column} — {feature_names[column]}"
        else:
            label = str(column)

        if i % 3 == 0:

            with col1:
                data[column] = st.number_input(
                    label,
                    value=0.0,
                    format="%.4f"
                )

        elif i % 3 == 1:

            with col2:
                data[column] = st.number_input(
                    label,
                    value=0.0,
                    format="%.4f"
                )

        else:

            with col3:
                data[column] = st.number_input(
                    label,
                    value=0.0,
                    format="%.4f"
                )

    st.markdown("")

    predict_button = st.form_submit_button(
        "🔮 Predict House Price",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # Create DataFrame in exact model column order
        input_data = pd.DataFrame(
            [data],
            columns=columns
        )

        # Make prediction
        prediction = model.predict(input_data)[0]

        st.markdown("---")

        st.header("💰 Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col2:

            st.metric(
                label="Estimated House Price",
                value=f"${prediction:,.2f}"
            )

        st.success(
            "✅ Prediction generated successfully!"
        )

        # Show entered values
        with st.expander("📋 View Input Values"):

            display_data = input_data.T

            display_data.columns = ["Value"]

            st.dataframe(
                display_data,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )


# ============================================================
# MACHINE LEARNING PIPELINE
# ============================================================

st.markdown("---")

st.header("🤖 Machine Learning Pipeline")

pipeline1, pipeline2, pipeline3, pipeline4 = st.columns(4)

with pipeline1:

    st.subheader("1️⃣ Data Preparation")

    st.write(
        "Data cleaning, missing-value handling, "
        "feature selection and preprocessing."
    )


with pipeline2:

    st.subheader("2️⃣ Feature Engineering")

    st.write(
        "Preparing and transforming input "
        "variables for machine learning."
    )


with pipeline3:

    st.subheader("3️⃣ Model Training")

    st.write(
        "Training and evaluating regression "
        "algorithms using performance metrics."
    )


with pipeline4:

    st.subheader("4️⃣ Deployment")

    st.write(
        "Deploying the trained model as an "
        "interactive Streamlit application."
    )


# ============================================================
# TECHNOLOGIES
# ============================================================

st.markdown("---")

st.header("🛠️ Technologies Used")

tech1, tech2, tech3, tech4, tech5 = st.columns(5)

with tech1:
    st.subheader("🐍")
    st.write("Python")

with tech2:
    st.subheader("📊")
    st.write("Pandas")

with tech3:
    st.subheader("🔢")
    st.write("NumPy")

with tech4:
    st.subheader("🤖")
    st.write("Scikit-learn")

with tech5:
    st.subheader("🚀")
    st.write("Streamlit")


# ============================================================
# PROJECT HIGHLIGHTS
# ============================================================

st.markdown("---")

st.header("⭐ Project Highlights")

highlight1, highlight2, highlight3 = st.columns(3)

with highlight1:

    st.write("✅ Linear Regression")
    st.write("✅ Outlier Detection")
    st.write("✅ Ridge Regression")
    st.write("✅ R² Score")

with highlight2:

    st.write("✅ Feature Engineering")
    st.write("✅ StandardScaler")
    st.write("✅ Lasso Regression")
    st.write("✅ Adjusted R²")

with highlight3:

    st.write("✅ Data Cleaning")
    st.write("✅ MinMaxScaler")
    st.write("✅ ElasticNet")
    st.write("✅ Model Deployment")


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("---")

st.info(
    "⚠️ This application provides an estimated house price "
    "based on the trained machine learning model. "
    "It should not be considered a professional property valuation."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "House Price Prediction • Machine Learning Project"
)

st.caption(
    "Built with Python • Scikit-learn • Streamlit"
)