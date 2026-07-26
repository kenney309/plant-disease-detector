import streamlit as st
import numpy as np
from PIL import Image
import keras

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# -----------------------------
# 38 PLANTVILLAGE CLASSES
# -----------------------------
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper_bell___Bacterial_spot",
    "Pepper_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return keras.saving.load_model(
        "hf://animeshakr/plant-disease-efficientnetv2s"
    )

# -----------------------------
# DISEASE INFORMATION
# -----------------------------
def get_advice(disease):

    if "healthy" in disease.lower():
        return (
            "The plant appears healthy. Continue good watering, "
            "proper sunlight, and regular monitoring."
        )

    if "Apple_scab" in disease:
        return (
            "Remove affected leaves and improve air circulation. "
            "Follow local agricultural guidance for appropriate fungicide use."
        )

    if "Late_blight" in disease:
        return (
            "Remove severely affected plant material and avoid "
            "watering foliage. Seek local agricultural advice."
        )

    if "Early_blight" in disease:
        return (
            "Remove affected leaves and maintain good field hygiene. "
            "Avoid overhead watering where possible."
        )

    if "Powdery_mildew" in disease:
        return (
            "Improve air circulation around plants and remove "
            "severely affected leaves."
        )

    return (
        "The image may show signs of plant disease. Remove severely "
        "affected material where appropriate and consult a local "
        "agricultural expert for specific treatment advice."
    )

# -----------------------------
# APP
# -----------------------------
st.title("🌿 Plant Disease Detector")

st.write(
    "Upload a clear image of a plant leaf. "
    "The AI model will analyze the image and predict "
    "the most likely PlantVillage disease category."
)

st.divider()

uploaded_file = st.file_uploader(
    "📷 Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )

    if st.button("🔍 Analyze Leaf"):

        with st.spinner("AI is analyzing the leaf..."):

            model = load_model()

            # EfficientNetV2S expects 384x384 input
            img = image.resize((384, 384))

            img_array = np.array(img)
            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            predictions = model.predict(
                img_array,
                verbose=0
            )

            predicted_index = int(
                np.argmax(predictions[0])
            )

            confidence = float(
                predictions[0][predicted_index]
            )

            predicted_class = CLASS_NAMES[
                predicted_index
            ]

        st.success("Analysis complete!")

        st.subheader("🔍 Diagnosis")

        st.write(
            f"**Prediction:** {predicted_class.replace('_', ' ')}"
        )

        st.write(
            f"**Confidence:** {confidence * 100:.2f}%"
        )

        st.progress(
            min(confidence, 1.0)
        )

        st.subheader("💡 Recommended Action")

        st.info(
            get_advice(predicted_class)
        )

        st.warning(
            "This AI result is an educational screening tool "
            "and should not replace advice from a qualified "
            "agricultural professional."
        )

else:

    st.info(
        "👆 Upload a clear image of a plant leaf to begin."
    )

st.divider()

st.subheader("📖 About the Project")

st.write(
    """
    The Plant Disease Detector is an Artificial Intelligence
    project designed to support smart agriculture.

    The system uses computer vision to analyze plant leaf images
    and classify them into PlantVillage disease categories.
    """
)
