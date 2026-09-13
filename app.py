import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB


st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)

st.title("🩺 Diabetes Prediction")
st.write("Predict diabetes using Gaussian Naive Bayes.")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("diabetes_prediction_dataset.csv")
    data = data.drop_duplicates()
    return data


data = load_data()

# Prepare encoders
gender_encoder = LabelEncoder()
smoking_encoder = LabelEncoder()

data["gender"] = gender_encoder.fit_transform(data["gender"])
data["smoking_history"] = smoking_encoder.fit_transform(
    data["smoking_history"]
)

X = data.drop("diabetes", axis=1)
y = data["diabetes"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = GaussianNB(var_smoothing=1.0)
model.fit(X_scaled, y)


st.subheader("Enter Patient Details")

gender = st.selectbox("Gender", ["Female", "Male", "Other"])
age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=30,
    step=1
)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_history = st.selectbox(
    "Smoking History",
    ["No Info", "never", "former", "current", "not current", "ever"]
)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
hba1c = st.number_input(
    "HbA1c Level",
    min_value=0.0,
    max_value=20.0,
    value=5.5
)
glucose = st.number_input(
    "Blood Glucose Level",
    min_value=0.0,
    max_value=500.0,
    value=100.0
)

if st.button("Predict Diabetes"):
    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose
    }])

    # Encode input using the same encoders
    input_data["gender"] = gender_encoder.transform(input_data["gender"])
    input_data["smoking_history"] = smoking_encoder.transform(
        input_data["smoking_history"]
    )

    # Scale input using the trained scaler
    input_scaled = scaler.transform(input_data)

    # Predict probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Use selected threshold
        # Use selected threshold
    threshold = 0.10
    prediction = int(probability >= threshold)

    st.subheader("Prediction Result")
    st.write(f"Predicted diabetes probability: **{probability:.2%}**")

    if prediction == 1:
        st.error("🔴 Prediction: Diabetic")
    else:
        st.success("🟢 Prediction: Non-Diabetic")

    st.warning(
        "This application is for educational purposes only and is not a medical diagnosis."
    )