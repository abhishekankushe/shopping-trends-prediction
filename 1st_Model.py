import streamlit as st
import pandas as pd
import pickle
import traceback

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="Shopping Trends - Subscription Predictor",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------
# Clean professional CSS
# -------------------------
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: #f4f6f8;
    }

    /* Main card */
    .card {
        background: #ffffff;
        padding: 26px;
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(16,24,40,0.06);
    }

    /* Heading styling */
    .title {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .subtitle {
        color: #475569;
        margin-bottom: 18px;
    }

    /* Primary button */
    .stButton>button {
        background: linear-gradient(90deg,#0ea5a5,#006b6b);
        color: #ffffff;
        font-weight: 600;
        padding: 10px 22px;
        border-radius: 8px;
        border: none;
        transform: translateZ(0);
        transition: transform .12s ease, box-shadow .12s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(6,95,95,0.12);
    }

    /* Result box */
    .result {
        background: #eefdfb;
        border-left: 4px solid #06b6d4;
        padding: 14px;
        border-radius: 8px;
        font-weight: 600;
        color: #064e57;
    }

    /* Small muted text */
    .muted {
        color: #6b7280;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Load artifacts (with friendly errors)
# -------------------------
try:
    best_knn = pickle.load(open("knn_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
    feature_order = pickle.load(open("feature_order.pkl", "rb"))
except FileNotFoundError as fnf:
    st.error("Required ML files are missing from the app directory. Please ensure the following files are present in the same folder as this script: knn_model.pkl, scaler.pkl, label_encoders.pkl, feature_order.pkl.")
    st.caption("Detailed error:")
    st.text(traceback.format_exc())
    st.stop()
except Exception as e:
    st.error("Failed to load model artifacts. Check server logs or verify the files.")
    st.caption("Detailed error:")
    st.text(traceback.format_exc())
    st.stop()

# -------------------------
# Page content
# -------------------------
container = st.container()
with container:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("<div class='title'>Shopping Trends — Subscription Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Provide a few key details to get a prediction. The model requires all features internally; missing ones are filled with defaults.</div>", unsafe_allow_html=True)

    # Two-column input layout
    col1, col2 = st.columns([1, 1])

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=30, step=1, format="%d")
        gender = st.selectbox("Gender", label_encoders["Gender"].classes_)
        category = st.selectbox("Category", label_encoders["Category"].classes_)

    with col2:
        purchase_amount = st.number_input("Purchase Amount (USD)", min_value=0.0, value=50.0, step=1.0, format="%.2f")
        payment_method = st.selectbox("Payment Method", label_encoders["Payment Method"].classes_)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    predict_btn = st.button("Predict")

    # After prediction
    if predict_btn:
        try:
            # Default placeholders for other model features (take first class or reasonable numeric)
            defaults = {
                "Item Purchased": label_encoders["Item Purchased"].classes_[0] if "Item Purchased" in label_encoders else "",
                "Location": label_encoders["Location"].classes_[0] if "Location" in label_encoders else "",
                "Size": label_encoders["Size"].classes_[0] if "Size" in label_encoders else "",
                "Color": label_encoders["Color"].classes_[0] if "Color" in label_encoders else "",
                "Season": label_encoders["Season"].classes_[0] if "Season" in label_encoders else "",
                "Review Rating": 3.5,
                "Shipping Type": label_encoders["Shipping Type"].classes_[0] if "Shipping Type" in label_encoders else "",
                "Discount Applied": label_encoders["Discount Applied"].classes_[0] if "Discount Applied" in label_encoders else "",
                "Promo Code Used": label_encoders["Promo Code Used"].classes_[0] if "Promo Code Used" in label_encoders else "",
                "Previous Purchases": 1,
                "Preferred Payment Method": label_encoders["Preferred Payment Method"].classes_[0] if "Preferred Payment Method" in label_encoders else "",
                "Frequency of Purchases": label_encoders["Frequency of Purchases"].classes_[0] if "Frequency of Purchases" in label_encoders else ""
            }

            # Build input dict (only the minimal 5 user inputs + defaults)
            input_data = {
                "Age": age,
                "Gender": gender,
                "Category": category,
                "Purchase Amount (USD)": purchase_amount,
                "Payment Method": payment_method,
            }
            input_data.update(defaults)

            new_data = pd.DataFrame([input_data])

            # Encode categorical features using saved LabelEncoders
            for col in new_data.select_dtypes(include="object").columns:
                if col not in label_encoders:
                    raise KeyError(f"Encoder for column '{col}' not found in label_encoders.")
                # transform expects array-like
                new_data[col] = label_encoders[col].transform(new_data[col])

            # Reorder columns to match training feature order and ensure all features present
            missing_cols = [c for c in feature_order if c not in new_data.columns]
            if missing_cols:
                raise ValueError(f"The following required features are missing from constructed input: {missing_cols}")

            new_data = new_data[feature_order]

            # Scale features using the saved scaler
            new_data_scaled = scaler.transform(new_data)

            # Make prediction
            pred = best_knn.predict(new_data_scaled)
            try:
                # If model supports predict_proba, show confidence
                prob = None
                if hasattr(best_knn, "predict_proba"):
                    proba = best_knn.predict_proba(new_data_scaled)
                    prob = proba.max(axis=1)[0]
            except Exception:
                prob = None

            # Decode predicted label (if present in encoders)
            pred_label = None
            if "Subscription Status" in label_encoders:
                pred_label = label_encoders["Subscription Status"].inverse_transform(pred)[0]
            else:
                pred_label = str(pred[0])

            # Display result in a styled box
            result_html = "<div class='result'>Prediction: {}{}</div>"
            prob_text = f" — Confidence: {prob:.2f}" if prob is not None else ""
            st.markdown(result_html.format(pred_label, prob_text), unsafe_allow_html=True)

            st.markdown("<div class='muted' style='margin-top:10px;'>Note: The app fills non-provided features with defaults chosen from training data. For best accuracy, retrain the model with fewer inputs or provide more fields.</div>", unsafe_allow_html=True)

        except Exception as err:
            st.error("An error occurred while making the prediction. See details below.")
            st.text(traceback.format_exc())

    st.markdown("</div>", unsafe_allow_html=True)
