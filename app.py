"""
Streamlit interface for the Natural Scene Image Classification project.

Run locally with:
    streamlit run app.py

Requires `best_model.keras` to be in the same folder as this script,
and .streamlit/config.toml for the theme.
"""

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import pillow_heif

# Registers HEIC/HEIF support with Pillow (iPhone photos use this format by default)
pillow_heif.register_heif_opener()

# --- Config (must match what the model was trained on) ---
IMG_SIZE = (150, 150)
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MODEL_PATH = "best_model.keras"

st.set_page_config(page_title="Scene Classifier", page_icon="🏔️", layout="centered")

# --- Custom CSS: rounded image corners + drop shadow on the prediction column ---
st.markdown(
    """
    <style>
    div[data-testid="stImage"] img {
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    }
    div[data-testid="column"]:nth-of-type(2) {
        background-color: #1A1D24;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict(image: Image.Image, model):
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    return predicted_class, confidence, predictions


# --- UI ---
st.title("🏔️ Natural Scene Classifier")
st.write(
    "Upload a photo and the model will classify it into one of six categories: "
    "**buildings, forest, glacier, mountain, sea, street**."
)

model = load_model()

uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error(
            "Couldn't read that file as an image. Make sure it's a valid photo "
            "(JPG, PNG, HEIC, WEBP, BMP, TIFF, GIF, etc.) and try again."
        )
        st.stop()

    with st.spinner("Classifying..."):
        predicted_class, confidence, predictions = predict(image, model)

    col1, col2 = st.columns(2, vertical_alignment="center")

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        st.metric(
            label="Top Prediction",
            value=predicted_class.capitalize(),
            delta=f"{confidence:.2f}% Confidence",
            delta_color="normal",
        )

    st.subheader("Class Probabilities")

    prob_df = pd.DataFrame(
        {"Class": CLASS_NAMES, "Probability": predictions}
    ).sort_values("Probability", ascending=True).set_index("Class")

    st.bar_chart(prob_df, horizontal=True, color="#2E7D32")

else:
    st.info("Upload an image above to get started.")
