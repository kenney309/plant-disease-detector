import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detector")

st.write(
    "Upload a clear image of a plant leaf. "
    "Our AI model will analyze it and predict "
    "a possible plant disease."
)

# 38 PlantVillage classes
CLASS_NAMES = [
    "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",
    "Blueberry - Healthy",
    "Cherry - Powdery Mildew",
    "Cherry - Healthy",
    "Corn - Cercospora Leaf Spot / Gray Leaf Spot",
    "Corn - Common Rust",
    "Corn - Northern Leaf Blight",
    "Corn - Healthy",
    "Grape - Black Rot",
    "Grape - Esca (Black Measles)",
    "Grape - Leaf Blight",
    "Grape - Healthy",
    "Orange - Huanglongbing (Citrus Greening)",
    "Peach - Bacterial Spot",
    "Peach - Healthy",
    "Pepper - Bacterial Spot",
    "Pepper - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Raspberry - Healthy",
    "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch",
    "Strawberry - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites",
    "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus",
    "Tomato - Healthy"
]

# Disease advice
def get_advice(disease):

    if "Healthy" in disease:
        return (
            "The plant appears healthy. Continue proper watering, "
            "adequate sunlight, good nutrition, and regular monitoring."
        )

    if "Early Blight" in disease:
        return (
            "Remove affected leaves and maintain good garden hygiene. "
            "Avoid watering the leaves directly."
        )

    if "Late Blight" in disease:
        return (
            "Remove severely affected plant material and avoid "
            "overhead watering. Seek advice from a local agricultural expert."
        )

    if "Powdery Mildew" in disease:
        return (
            "Improve air circulation around the plant and remove "
            "severely affected leaves."
        )

    if "Bacterial Spot" in disease:
        return (
            "Remove badly affected leaves and avoid working with plants "
            "when they are wet. Consult an agricultural professional "
            "for appropriate treatment."
        )

    if "Apple Scab" in disease:
        return (
            "Remove infected leaves and fruit where practical and "
            "improve air circulation around the tree."
        )

    return (
        "The image may show signs of plant disease. Remove severely "
        "affected plant material where appropriate and consult a "
        "qualified agricultural professional for specific treatment."
    )


# Upload image
uploaded_file = st.file_uploader(
    "📷 Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Plant Leaf",
        use_container_width=True
    )

    if st.button("🔍 ANALYZE LEAF"):

        st.info(
            "The image has been received. "
            "The AI model is ready for prediction."
        )

        # Temporary prediction test
        # The actual trained model will be connected here.
        st.subheader("🌿 AI Analysis")

        st.success(
            "Image successfully processed!"
        )

        st.write(
            "The AI prediction engine is now ready "
            "for the trained PlantVillage model."
        )

        st.info(
            "Next step: connect the trained 38-class "
            "PlantVillage model for actual disease predictions."
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

    The system analyzes images of plant leaves and is designed
    to identify possible plant diseases and provide useful
    information about plant health.
    """
)

st.caption(
    "🌿 Plant Disease Detector | AI for Smart Agriculture"
)
