"""
Streamlit interface for the Natural Scene Image Classification project.

Run locally with:
    streamlit run app.py

Requires `best_model.keras` to be in the same folder as this script.
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --- Config (must match what the model was trained on) ---
IMG_SIZE = (150, 150)
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MODEL_PATH = "best_model.keras"

st.set_page_config(page_title="Scene Classifier", page_icon="🏔️", layout="centered")


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

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Classifying..."):
        predicted_class, confidence, predictions = predict(image, model)

    with col2:
        st.subheader("Prediction")
        st.markdown(f"### {predicted_class}")
        st.metric("Confidence", f"{confidence:.2f}%")

        st.write("**Class probabilities:**")
        for cls, prob in sorted(
            zip(CLASS_NAMES, predictions), key=lambda x: -x[1]
        ):
            st.write(f"{cls}")
            st.progress(float(prob))
else:
    st.info("Upload an image above to get started.")
