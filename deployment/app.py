
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os


# Load model function
@st.cache_resource
def load_model():
    try:
        # Try to load from HuggingFace Hub
        model_path = hf_hub_download(
            repo_id="balakishan77/Tourism_Package",
            filename="best_tourism_package_model_v1.joblib",
            repo_type="model"
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Main header
st.title(" Tourism Package Prediction")

# Set page config
st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Subtitle
st.markdown("** Predict whether a customer will purchase the Wellness🌴 Tourism ✈️ Package based on their profile and preferences.")
st.markdown("---")

# Load model
model = load_model()
if model is None:
    st.error("Sorry, an issue occurred during loading. Please check the model availability.")
    st.stop()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Profile")

    age = st.slider("Age", min_value=18, max_value=80, value=35, help="Customer's age")

    gender = st.selectbox("Gender", ["Male", "Female"], help="Customer's gender")

    occupation = st.selectbox("Occupation", [
        "Salaried", "Small_Business", "Large_Business", "Free_Lancer"
    ], help="Customer's occupation type")

    designation = st.selectbox("Designation", [
        "Executive", "Manager", "Senior_Manager", "AVP", "VP"
    ], help="Job designation in organization")

    monthly_income = st.number_input(
        "Monthly Income (₹)", 
        min_value=10000, 
        max_value=500000, 
        value=50000, 
        step=5000,
        help="Gross monthly income"
    )

    marital_status = st.selectbox("Marital Status", [
        "Single", "Married", "Divorced", "Unmarried"
    ], help="Customer's marital status")

    city_tier = st.selectbox("City Tier", [
        "Tier_1", "Tier_2", "Tier_3"
    ], help="City development level (Tier 1 > Tier 2 > Tier 3)")

with col2:
    st.subheader("🧳 Preferences")

    number_of_person_visiting = st.slider(
        "Number of People Visiting", 
        min_value=1, 
        max_value=10, 
        value=2,
        help="Total people accompanying"
    )

    number_of_trips = st.slider(
        "Average Annual Trips", 
        min_value=0, 
        max_value=25, 
        value=3,
        help="Average trips per year"
    )

    preferred_property_star = st.selectbox("Preferred Hotel Star Rating", [
        "3.0", "4.0", "5.0"
    ], help="Preferred hotel rating")

    passport = st.selectbox("Passport Status", [
        "Yes", "No"
    ], help="Valid passport availability")

    own_car = st.selectbox("Own Car", [
        "Yes", "No"
    ], help="Car ownership status")

    number_of_children_visiting = st.slider(
        "Children Visiting (Under 5)", 
        min_value=0, 
        max_value=5, 
        value=0,
        help="Number of children under 5"
    )

    st.subheader("🤝 Wellness Package Interaction")

    type_of_contact = st.selectbox("Contact Method", [
        "Company_Invited", "Self_Inquiry"
    ], help="How customer was contacted")

    duration_of_pitch = st.slider(
        "Pitch Duration (minutes)", 
        min_value=5, 
        max_value=120, 
        value=30,
        help="Sales presentation duration"
    )

    product_pitched = st.selectbox("Product Pitched", [
        "Basic", "Standard", "Deluxe", "Super_Deluxe", "King"
    ], help="Type of package offered")

    pitch_satisfaction_score = st.slider(
        "Pitch Satisfaction Score", 
        min_value=1, 
        max_value=5, 
        value=3,
        help="Customer satisfaction with sales pitch"
    )

    number_of_followups = st.slider(
        "Number of Follow-ups", 
        min_value=0, 
        max_value=10, 
        value=2,
        help="Sales team follow-up attempts"
    )

# Prediction button
st.markdown("")

col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
with col_btn2:
    predict_button = st.button("🔮 Predict Purchase Likelihood", use_container_width=True)

if predict_button:
    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'DurationOfPitch': [duration_of_pitch],
        'MonthlyIncome': [monthly_income],
        'TypeofContact': [type_of_contact],
        'CityTier': [city_tier],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [number_of_person_visiting],
        'NumberOfFollowups': [number_of_followups],
        'ProductPitched': [product_pitched],
        'PreferredPropertyStar': [float(preferred_property_star)],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [number_of_trips],
        'Passport': [1 if passport == "Yes" else 0],
        'PitchSatisfactionScore': [pitch_satisfaction_score],
        'OwnCar': [1 if own_car == "Yes" else 0],
        'NumberOfChildrenVisiting': [number_of_children_visiting],
        'Designation': [designation]
    })

    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Display results
        st.markdown("")

        if prediction == 1:
            st.success("🎉 **HIGH LIKELIHOOD TO PURCHASE!** 🎉")
            st.info(f"**Confidence:** {prediction_proba[1]:.1%}")
            st.success("✅ **Recommendation**: This customer is likely to purchase the Wellness Tourism Package. Proceed with the sales process!")

        else:
            st.error("❌ **LOW LIKELIHOOD TO PURCHASE**")
            st.info(f"**Confidence:** {prediction_proba[0]:.1%}")
            st.warning("⚠️ **Recommendation**: This customer may not purchase the package. Consider adjusting the approach or offering alternatives.")

        # Show probability breakdown
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            st.metric("Won't Purchase Probability", f"{prediction_proba[0]:.1%}")
        with col_prob2:
            st.metric("Will Purchase Probability", f"{prediction_proba[1]:.1%}")

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("**Tourism Package Predictor** | Powered by Machine Learning | Built with Streamlit 🚀")
st.caption("Developed for Tourism Package Project")

