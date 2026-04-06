# app.py
import streamlit as st
from predict import predict_gesture
from PIL import Image
import time

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Hand Gesture Recognition (Pattern Recognition)",
    page_icon="🖐️",
    layout="centered"
)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("📌 Project Overview")
st.sidebar.markdown(
    """
    **Project Type:** Pattern Recognition  
    **Input:** Static Images  
    **Dataset:** HG14  
    **Features:** Multi-scale HOG  
    **Classifier:** PCA + Random Forest  
    **Accuracy:** ~75%
    """
)

# -------------------------------
# MAIN TITLE
# -------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🖐️ Hand Gesture Recognition System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center; color:gray;'>Static Image-Based Pattern Recognition Approach</h4>",
    unsafe_allow_html=True
)

st.write("---")

# -------------------------------
# ABOUT PROJECT
# -------------------------------
with st.expander("📘 About the Project", expanded=False):
    st.markdown(
        """
        This application demonstrates **hand gesture recognition on static images**
        using **classical Pattern Recognition techniques**.

        ### Methodology:
        - Image Preprocessing
        - Multi-scale Histogram of Oriented Gradients (HOG)
        - Principal Component Analysis (PCA)
        - Random Forest Classification

        The system is trained and evaluated under **controlled conditions** using
        the **HG14 dataset**.
        """
    )

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
st.markdown("### 📤 Upload a Gesture Image")

uploaded_file = st.file_uploader(
    "Upload an image from the dataset or a similar controlled setup",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Gesture Image", use_column_width=True)

    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔍 Extracting features and recognizing gesture..."):
        time.sleep(1)
        prediction = predict_gesture("temp.jpg")

    st.success("✅ Gesture Recognized")

    st.markdown(
        f"""
        <div style="
            background-color:#f0f2f6;
            padding:20px;
            border-radius:10px;
            text-align:center;
            font-size:22px;
            font-weight:bold;">
            Predicted Gesture: <span style="color:#2c7be5;">{prediction}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# LIMITATIONS (VERY IMPORTANT)
# -------------------------------
with st.expander("⚠️ Limitations & Future Scope", expanded=False):
    st.markdown(
        """
        **Limitations:**
        - Designed for static images under controlled conditions
        - Performance degrades on live webcam images due to domain shift
        - Sensitive to background and lighting variations

        **Future Scope:**
        - Integrate hand detection and landmark-based features
        - Extend system to real-time gesture recognition
        - Use deep learning models for improved robustness
        """
    )

# -------------------------------
# FOOTER
# -------------------------------
st.write("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Pattern Recognition Mini Project | Static Image-Based Gesture Recognition</p>",
    unsafe_allow_html=True
)
