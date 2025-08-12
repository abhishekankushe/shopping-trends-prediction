import streamlit as st
import pandas as pd
import pickle

# --- Page Config ---
st.set_page_config(
    page_title="Shopping Trends Subscription Prediction",
    page_icon="🛍️",
    layout="centered",
)

# --- Load Model ---
best_knn = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# --- Custom CSS ---
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 8px 20px;
        }
        .stButton>button:hover {
            background-color: #ff3333;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🛒 Shopping Trends Subscription Prediction")
st.markdown("Fill in the customer details to **predict their subscription status**.")

# --- Form Layout ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("👤 Age", min_value=10, max_value=100, value=30)
        gender = st.selectbox("🚻 Gender", label_encoders['Gender'].classes_)
        item_purchased = st.selectbox("🛍️ Item Purchased", label_encoders['Item Purchased'].classes_)
        category = st.selectbox("📦 Category", label_encoders['Category'].classes_)
        purchase_amount = st.number_input("💰 Purchase Amount (USD)", min_value=0, value=50)
        location = st.selectbox("📍 Location", label_encoders['Location'].classes_)
        size = st.selectbox("📏 Size", label_encoders['Size'].classes_)
        color = st.selectbox("🎨 Color", label_encoders['Color'].classes_)
        season = st.selectbox("🌤️ Season", label_encoders['Season'].classes_)

    with col2:
        review_rating = st.number_input("⭐ Review Rating", min_value=0.0, max_value=5.0, step=0.1, value=3.5)
        payment_method = st.selectbox("💳 Payment Method", label_encoders['Payment Method'].classes_)
        shipping_type = st.selectbox("📦 Shipping Type", label_encoders['Shipping Type'].classes_)
        discount_applied = st.selectbox("🏷️ Discount Applied", label_encoders['Discount Applied'].classes_)
        promo_code_used = st.selectbox("🎁 Promo Code Used", label_encoders['Promo Code Used'].classes_)
        previous_purchases = st.number_input("🛒 Previous Purchases", min_value=0, value=5)
        preferred_payment_method = st.selectbox("💵 Preferred Payment Method", label_encoders['Preferred Payment Method'].classes_)
        frequency_of_purchases = st.selectbox("📆 Frequency of Purchases", label_encoders['Frequency of Purchases'].classes_)

    submitted = st.form_submit_button("🔍 Predict Subscription Status")

    if submitted:
        # Prepare DataFrame
        new_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Item Purchased': item_purchased,
            'Category': category,
            'Purchase Amount (USD)': purchase_amount,
            'Location': location,
            'Size': size,
            'Color': color,
            'Season': season,
            'Review Rating': review_rating,
            'Payment Method': payment_method,
            'Shipping Type': shipping_type,
            'Discount Applied': discount_applied,
            'Promo Code Used': promo_code_used,
            'Previous Purchases': previous_purchases,
            'Preferred Payment Method': preferred_payment_method,
            'Frequency of Purchases': frequency_of_purchases
        }])

        # Encode categorical features
        for col in new_data.select_dtypes(include='object').columns:
            new_data[col] = label_encoders[col].transform(new_data[col])

        # Scale numerical features
        new_data_scaled = scaler.transform(new_data)

        # Predict
        pred = best_knn.predict(new_data_scaled)
        pred_label = label_encoders['Subscription Status'].inverse_transform(pred)[0]

        # Display Result
        st.markdown(
            f"<div style='background-color:#d4edda;padding:15px;border-radius:8px;font-size:18px;'>"
            f"<b>✅ Predicted Subscription Status:</b> {pred_label}</div>",
            unsafe_allow_html=True
        )
