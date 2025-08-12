import streamlit as st
import pandas as pd
import pickle
import os
import traceback

# --- Page Config ---
st.set_page_config(
    page_title="Shopping Trends Prediction",
    page_icon="🛍️",
    layout="centered"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .main {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 24px;
            transition: 0.3s;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #e63e3e;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# --- Debug: show files in the directory ---
st.write("📂 Files in app directory:", os.listdir())

# --- Load Model with Error Handling ---
try:
    best_knn = pickle.load(open("knn_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Failed to load model/scaler/encoders: {e}")
    st.text(traceback.format_exc())
    st.stop()

# --- Title ---
st.title("🛒 Shopping Trends Subscription Prediction")
st.markdown("Fill in the details below to predict **Subscription Status**.")

# --- Input Layout ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", min_value=10, max_value=100, value=30)
    gender = st.selectbox("🚻 Gender", label_encoders['Gender'].classes_)
    category = st.selectbox("📦 Category", label_encoders['Category'].classes_)

with col2:
    purchase_amount = st.number_input("💰 Purchase Amount (USD)", min_value=0, value=50)
    payment_method = st.selectbox("💳 Payment Method", label_encoders['Payment Method'].classes_)

# --- Prediction Button ---
if st.button("🔍 Predict Subscription Status"):
    try:
        # Fill missing features with defaults
        default_values = {
            'Item Purchased': label_encoders['Item Purchased'].classes_[0],
            'Location': label_encoders['Location'].classes_[0],
            'Size': label_encoders['Size'].classes_[0],
            'Color': label_encoders['Color'].classes_[0],
            'Season': label_encoders['Season'].classes_[0],
            'Review Rating': 3.5,
            'Shipping Type': label_encoders['Shipping Type'].classes_[0],
            'Discount Applied': label_encoders['Discount Applied'].classes_[0],
            'Promo Code Used': label_encoders['Promo Code Used'].classes_[0],
            'Previous Purchases': 1,
            'Preferred Payment Method': label_encoders['Preferred Payment Method'].classes_[0],
            'Frequency of Purchases': label_encoders['Frequency of Purchases'].classes_[0]
        }

        # Create DataFrame with all required features
        input_data = {
            'Age': age,
            'Gender': gender,
            'Category': category,
            'Purchase Amount (USD)': purchase_amount,
            'Payment Method': payment_method
        }
        input_data.update(default_values)

        new_data = pd.DataFrame([input_data])

        # Encode categorical features
        for col in new_data.select_dtypes(include='object').columns:
            new_data[col] = label_encoders[col].transform(new_data[col])

        # Scale numerical features
        new_data_scaled = scaler.transform(new_data)

        # Predict
        pred = best_knn.predict(new_data_scaled)
        pred_label = label_encoders['Subscription Status'].inverse_transform(pred)[0]

        # Styled result box
        st.markdown(
            f"<div style='background-color:#d4edda;padding:15px;border-radius:8px;font-size:18px;'>"
            f"✅ <b>Predicted Subscription Status:</b> {pred_label}</div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.text(traceback.format_exc())
