import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os
import base64

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)


# =========================================================
# BACKGROUND IMAGE
# =========================================================

def add_background(image_file):

    if os.path.exists(image_file):

        with open(image_file, "rb") as file:

            encoded = base64.b64encode(
                file.read()
            ).decode()

        st.markdown(
            f"""
            <style>

            .stApp {{
                background-image:
                linear-gradient(
                    rgba(255,255,255,0.82),
                    rgba(255,255,255,0.82)
                ),
                url("data:image/jpeg;base64,{encoded}");

                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

            </style>
            """,
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "Background image not found. "
            "Make sure background.jpg is in "
            "the same folder as app.py."
        )


add_background(
    "background.jpg"
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "🌿 Plant Disease Detector"
)

st.write(
    "Upload a clear image of a plant leaf "
    "and the AI will analyze it."
)

st.divider()


# =========================================================
# MODEL SETTINGS
# =========================================================

MODEL_URL = (
    "https://huggingface.co/animeshakr/"
    "plant-disease-efficientnetv2s/resolve/main/"
    "model_float16_quant.tflite"
)

MODEL_PATH = (
    "plant_disease_model.tflite"
)


# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = [

    "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",

    "Blueberry - Healthy",

    "Cherry - Powdery Mildew",
    "Cherry - Healthy",

    "Corn - Common Rust",
    "Corn - Northern Leaf Blight",
    "Corn - Healthy",

    "Grape - Black Rot",
    "Grape - Esca",
    "Grape - Leaf Blight",
    "Grape - Healthy",

    "Orange - Citrus Greening",

    "Peach - Bacterial Spot",
    "Peach - Healthy",

    "Pepper - Bacterial Spot",
    "Pepper - Healthy",

    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",

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
# LOAD MODEL
# =========================================================

@st.cache_resource
def get_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        with st.spinner(
            "Downloading AI model..."
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

    interpreter = (
        tf.lite.Interpreter(
            model_path=MODEL_PATH
        )
    )

    interpreter.allocate_tensors()

    inputs = (
        interpreter.get_input_details()
    )

    outputs = (
        interpreter.get_output_details()
    )

    return (
        interpreter,
        inputs,
        outputs
    )


# =========================================================
# AI PREDICTION
# =========================================================

def analyze_leaf(image):

    (
        interpreter,
        inputs,
        outputs
    ) = get_model()

    shape = inputs[0]["shape"]

    height = shape[1]

    width = shape[2]

    image = image.resize(
        (
            width,
            height
        )
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = (
        image_array / 255.0
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    interpreter.set_tensor(
        inputs[0]["index"],
        image_array
    )

    interpreter.invoke()

    prediction = (
        interpreter.get_tensor(
            outputs[0]["index"]
        )[0]
    )

    top = np.argsort(
        prediction
    )[::-1][:3]

    results = []

    for index in top:

        if index < len(
            CLASS_NAMES
        ):

            name = (
                CLASS_NAMES[index]
            )

        else:

            name = "Unknown"

        confidence = float(
            prediction[index]
        )

        results.append(
            (
                name,
                confidence
            )
        )

    return results


# =========================================================
# PLANT SELECTION
# =========================================================

plant = st.selectbox(
    "🌱 Select the plant:",
    [
        "Apple",
        "Banana",
        "Grape",
        "Guava",
        "Mango",
        "Tomato",
        "Other"
    ]
)


# =========================================================
# IMAGE INPUT
# =========================================================

st.subheader(
    "📷 Choose Your Leaf Image"
)

input_method = st.radio(
    "Select an option:",
    [
        "Upload Image",
        "Use Camera"
    ],
    horizontal=True
)


uploaded_file = None


if input_method == "Upload Image":

    uploaded_file = (
        st.file_uploader(
            "Upload a leaf image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )
    )

else:

    uploaded_file = (
        st.camera_input(
            "Take a picture of the leaf"
        )
    )


# =========================================================
# ANALYSIS
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )


    if plant in [
        "Guava",
        "Mango",
        "Banana"
    ]:

        st.warning(
            f"⚠️ The current AI model was "
            f"not specifically trained on "
            f"{plant}. The result may be inaccurate."
        )


    if st.button(
        "🔍 ANALYZE LEAF",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "🤖 Analyzing leaf..."
            ):

                results = (
                    analyze_leaf(
                        image
                    )
                )


            prediction = (
                results[0][0]
            )

            confidence = (
                results[0][1]
            )


            st.success(
                "Analysis complete!"
            )

            st.divider()


            # =============================================
            # MAIN RESULT
            # =============================================

            st.subheader(
                "🌿 AI Result"
            )

            st.write(
                f"**Plant Selected:** "
                f"{plant}"
            )

            st.write(
                f"**Prediction:** "
                f"{prediction}"
            )

            st.write(
                f"**Confidence:** "
                f"{confidence * 100:.2f}%"
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


            # =============================================
            # CONFIDENCE WARNING
            # =============================================

            if confidence < 0.40:

                st.error(
                    """
                    ⚠️ LOW CONFIDENCE

                    The AI is not confident about
                    this result. The prediction may
                    be incorrect.
                    """
                )

            elif confidence < 0.70:

                st.warning(
                    """
                    ⚠️ MODERATE CONFIDENCE

                    Consider checking the result
                    with an agricultural expert.
                    """
                )

            else:

                st.success(
                    "✅ Higher confidence prediction."
                )


            # =============================================
            # TOP 3 PREDICTIONS
            # =============================================

            st.subheader(
                "🔎 Top 3 Predictions"
            )

            for number, result in enumerate(
                results,
                start=1
            ):

                name = result[0]

                score = result[1]

                st.write(
                    f"{number}. **{name}** — "
                    f"{score * 100:.2f}%"
                )


            # =============================================
            # RECOMMENDATION
            # =============================================

            st.subheader(
                "💡 Recommendation"
            )


            if "Healthy" in prediction:

                st.success(
                    """
                    🌿 The plant appears healthy
                    according to the AI prediction.

                    Recommended actions:

                    • Continue regular plant care.
                    • Monitor the plant regularly.
                    • Maintain good sanitation.
                    • Provide appropriate water
                      and nutrients.
                    """
                )


            elif confidence < 0.40:

                st.info(
                    """
                    The AI result is uncertain.

                    Take another clear photograph
                    in good lighting.

                    Make sure the leaf is clearly
                    visible and not blurry.

                    For a confirmed diagnosis,
                    consult an agricultural expert.
                    """
                )


            else:

                st.warning(
                    f"""
                    The AI detected a possible
                    condition:

                    {prediction}

                    Recommended actions:

                    • Monitor the plant closely.
                    • Remove severely affected leaves
                      when appropriate.
                    • Keep the growing area clean.
                    • Improve air circulation.
                    • Avoid unnecessary leaf wetness.
                    • Consult an agricultural expert
                      for confirmed diagnosis and treatment.
                    """
                )


            # =============================================
            # DOWNLOAD REPORT
            # =============================================

            st.subheader(
                "📄 Analysis Report"
            )

            report = f"""
PLANT DISEASE DETECTOR
======================

Selected Plant:
{plant}

AI Prediction:
{prediction}

Confidence:
{confidence * 100:.2f}%

TOP 3 PREDICTIONS
=================

1. {results[0][0]}
Confidence: {results[0][1] * 100:.2f}%

2. {results[1][0]}
Confidence: {results[1][1] * 100:.2f}%

3. {results[2][0]}
Confidence: {results[2][1] * 100:.2f}%


RECOMMENDATION
==============

The AI result is a supporting prediction
and should not replace professional
agricultural diagnosis.

If the confidence is low, take another
clear image or consult an agricultural expert.


PLANT DISEASE DETECTOR
AI FOR SMART AGRICULTURE
"""

            st.download_button(
                "📥 Download Analysis Report",
                report,
                file_name=(
                    "plant_analysis_report.txt"
                ),
                mime="text/plain",
                use_container_width=True
            )


        except Exception as error:

            st.error(
                "❌ Something went wrong."
            )

            st.write(
                str(error)
            )


else:

    st.info(
        "📷 Upload or capture a plant leaf "
        "image to start."
    )


# =========================================================
# ABOUT THE PROJECT
# =========================================================

st.divider()

st.subheader(
    "📖 About the Project"
)

st.write(
    """
    The Plant Disease Detector is an AI-based
    smart agriculture project.

    It analyzes images of plant leaves and provides
    possible disease predictions, confidence scores,
    and general recommendations.

    The system is designed as a supporting tool
    and should not replace professional agricultural
    advice.
    """
)

st.caption(
    "🌿 Plant Disease Detector | AI for Smart Agriculture"
)
