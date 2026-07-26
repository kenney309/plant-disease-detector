import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os
from datetime import datetime


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Smart Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("🌿 Smart Plant Disease Detector")

st.write(
    "Upload a clear plant leaf image. "
    "The AI will analyze the image and provide "
    "a possible disease prediction."
)

st.info(
    "💡 For best results, use a clear, close-up photo "
    "of one leaf with good lighting."
)

st.divider()


# =========================================================
# MODEL
# =========================================================

MODEL_URL = (
    "https://huggingface.co/animeshakr/"
    "plant-disease-efficientnetv2s/resolve/main/"
    "model_float16_quant.tflite"
)

MODEL_PATH = "plant_disease_model.tflite"


# =========================================================
# OFFICIAL MODEL CLASS MAPPING
# This order matches the model's class_indices.json
# =========================================================

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


# =========================================================
# PLANTS SUPPORTED BY THE MODEL
# =========================================================

SUPPORTED_PLANTS = [
    "Apple",
    "Blueberry",
    "Cherry",
    "Corn",
    "Grape",
    "Orange",
    "Peach",
    "Pepper",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato"
]


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    # Download model if it does not exist
    if not os.path.exists(MODEL_PATH):

        with st.spinner(
            "Downloading AI model for the first time..."
        ):

            response = requests.get(
                MODEL_URL,
                timeout=300
            )

            response.raise_for_status()

            with open(
                MODEL_PATH,
                "wb"
            ) as file:

                file.write(
                    response.content
                )

    # Load TFLite model
    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    return (
        interpreter,
        input_details,
        output_details
    )


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_leaf(image):

    (
        interpreter,
        input_details,
        output_details
    ) = load_model()


    # -----------------------------------------------------
    # MODEL EXPECTS 384 x 384
    # -----------------------------------------------------

    image = image.resize(
        (384, 384)
    )


    # -----------------------------------------------------
    # CONVERT TO FLOAT32
    # -----------------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )


    # -----------------------------------------------------
    # IMPORTANT
    # The model card's TFLite example uses raw float32
    # pixel values, not /255 normalization.
    # -----------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # -----------------------------------------------------
    # RUN MODEL
    # -----------------------------------------------------

    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )

    interpreter.invoke()


    # -----------------------------------------------------
    # GET OUTPUT
    # -----------------------------------------------------

    predictions = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]


    # -----------------------------------------------------
    # TOP 5 RESULTS
    # -----------------------------------------------------

    top_indices = np.argsort(
        predictions
    )[::-1][:5]


    results = []

    for index in top_indices:

        if index < len(CLASS_NAMES):

            results.append(
                {
                    "class": CLASS_NAMES[index],
                    "confidence": float(
                        predictions[index]
                    )
                }
            )

    return results


# =========================================================
# SELECT IMAGE
# =========================================================

st.subheader(
    "📷 Upload Your Leaf"
)

input_method = st.radio(
    "Choose how to provide the image:",
    [
        "Upload Image",
        "Use Camera"
    ],
    horizontal=True
)


uploaded_file = None


if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose a clear leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

else:

    uploaded_file = st.camera_input(
        "Take a picture of the leaf"
    )


# =========================================================
# ANALYSIS
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # Show image
    st.image(
        image,
        caption="Leaf Image",
        use_container_width=True
    )


    st.divider()


    if st.button(
        "🔍 ANALYZE LEAF",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "🤖 AI is analyzing the leaf..."
            ):

                results = predict_leaf(
                    image
                )


            # -------------------------------------------------
            # BEST RESULT
            # -------------------------------------------------

            best_result = results[0]

            prediction = (
                best_result["class"]
            )

            confidence = (
                best_result["confidence"]
            )


            analysis_time = (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )


            # -------------------------------------------------
            # DISPLAY RESULT
            # -------------------------------------------------

            st.success(
                "✅ Analysis Complete"
            )


            st.subheader(
                "🌿 AI Prediction"
            )


            st.write(
                f"**Prediction:** {prediction}"
            )

            st.write(
                f"**Confidence:** "
                f"{confidence * 100:.2f}%"
            )

            st.write(
                f"**Analysis Time:** "
                f"{analysis_time}"
            )


            st.progress(
                min(
                    max(
                        confidence,
                        0.0
                    ),
                    1.0
                )
            )


            # =================================================
            # CONFIDENCE HANDLING
            # =================================================

            if confidence < 0.40:

                st.error(
                    """
                    ⚠️ LOW CONFIDENCE

                    The AI is not confident enough to
                    give a reliable disease diagnosis.

                    The result shown above should NOT be
                    treated as a confirmed diagnosis.

                    Please take a clearer close-up image
                    with good lighting.
                    """
                )


            elif confidence < 0.70:

                st.warning(
                    """
                    ⚠️ MODERATE CONFIDENCE

                    The AI has some uncertainty.

                    Consider taking another image or
                    consulting an agricultural expert.
                    """
                )


            else:

                st.success(
                    "✅ The AI has higher confidence "
                    "in this prediction."
                )


            # =================================================
            # TOP 5 RESULTS
            # =================================================

            st.divider()

            st.subheader(
                "🔎 Top AI Predictions"
            )


            for number, result in enumerate(
                results,
                start=1
            ):

                st.write(
                    f"**{number}. "
                    f"{result['class']}** — "
                    f"{result['confidence'] * 100:.2f}%"
                )


            # =================================================
            # RECOMMENDATION
            # =================================================

            st.divider()

            st.subheader(
                "💡 Recommendation"
            )


            if confidence < 0.40:

                st.info(
                    """
                    🌱 The AI cannot confidently identify
                    the condition in this image.

                    Recommended actions:

                    • Take another clear photograph.
                    • Use natural daylight.
                    • Photograph one leaf at a time.
                    • Make sure the leaf fills much of
                      the image.
                    • Avoid blurry images.
                    • Consult an agricultural expert
                      for confirmation.
                    """
                )


            elif "Healthy" in prediction:

                st.success(
                    """
                    🌿 The AI predicts that the leaf
                    may be healthy.

                    Continue monitoring the plant and
                    maintain good agricultural practices.
                    """
                )


            else:

                st.warning(
                    f"""
                    🌱 Possible condition:

                    {prediction}

                    This is an AI prediction, not a
                    confirmed diagnosis.

                    Monitor the plant and seek advice
                    from an agricultural professional
                    before taking treatment actions.
                    """
                )


            # =================================================
            # DOWNLOAD REPORT
            # =================================================

            st.divider()

            st.subheader(
                "📄 Analysis Report"
            )


            report = f"""
SMART PLANT DISEASE DETECTOR
================================

AI PREDICTION
--------------------------------

Prediction:
{prediction}

Confidence:
{confidence * 100:.2f}%

Analysis Time:
{analysis_time}


TOP 5 PREDICTIONS
--------------------------------

"""


            for number, result in enumerate(
                results,
                start=1
            ):

                report += (
                    f"{number}. "
                    f"{result['class']} - "
                    f"{result['confidence'] * 100:.2f}%\n"
                )


            report += """

IMPORTANT NOTICE
--------------------------------

This AI system provides a supporting prediction.
It does not replace professional agricultural
diagnosis.

The model was trained using PlantVillage images
and may perform differently on real-world field
photographs.

For important crop management decisions,
consult a qualified agricultural professional.

SMART PLANT DISEASE DETECTOR
AI FOR SMART AGRICULTURE
"""


            st.download_button(
                "📥 Download Report",
                report,
                file_name=(
                    "plant_disease_report.txt"
                ),
                mime="text/plain",
                use_container_width=True
            )


        except Exception as error:

            st.error(
                "❌ An error occurred during analysis."
            )

            st.write(
                str(error)
            )


else:

    st.info(
        "📷 Upload or capture a leaf image "
        "to begin."
    )


# =========================================================
# ABOUT
# =========================================================

st.divider()

st.subheader(
    "📖 About the Project"
)

st.write(
    """
    The Smart Plant Disease Detector is an
    Artificial Intelligence project designed
    to support smart agriculture.

    It analyzes plant leaf images and provides
    possible disease classifications and confidence
    scores.

    The system is intended as a supporting tool
    and should not replace professional agricultural
    diagnosis.
    """
)


st.caption(
    "🌿 Smart Plant Disease Detector | "
    "AI for Smart Agriculture"
)
