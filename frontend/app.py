import streamlit as st
import requests
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🕵️",
    layout="centered"
)

# -------------------------------
# UI
# -------------------------------
st.title("🕵️ Deepfake Image Detector")
st.write("Upload an image to check whether it is **REAL** or **DEEPFAKE**.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing image..."):
            response = requests.post(
                API_URL,
                files={"file": uploaded_file.getvalue()}
            )

        if response.status_code == 200:
            result = response.json()

            label = result["status"]
            confidence = result["confidence"]

            if label == "DEEPFAKE":
                st.error(f"🚨 DEEPFAKE detected ({confidence}%)")
            else:
                st.success(f"✅ REAL image ({confidence}%)")

        else:
            st.error("❌ Error contacting the API.")
