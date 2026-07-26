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
    "Upload or capture a clear image of a plant leaf. "
    "The AI will analyze the image and provide a possible "
    "plant disease prediction, confidence score and advice."
)

st.info(
    "📸 For best results, use a clear close-up image "
    "of one leaf with good lighting."
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

MODEL_PATH = "plant_disease_model.tflite"


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
# LOAD AI MODEL
# =========================================================

@st.cache_resource
def load_model():

    # Download model if it is not already available
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


    # Load TensorFlow Lite model
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
# AI PREDICTION FUNCTION
# =========================================================

def predict_leaf(image):

    (
        interpreter,
        input_details,
        output_details
    ) = load_model()


    # -----------------------------------------------------
    # GET MODEL INPUT SIZE
    # -----------------------------------------------------

    input_shape = (
        input_details[0]["shape"]
    )

    height = int(
        input_shape[1]
    )

    width = int(
        input_shape[2]
    )


    # -----------------------------------------------------
    # RESIZE IMAGE
    # -----------------------------------------------------

    image = image.resize(
        (
            width,
            height
        )
    )


    # -----------------------------------------------------
    # CONVERT IMAGE TO NUMPY
    # -----------------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )


    # -----------------------------------------------------
    # ADD BATCH DIMENSION
    # -----------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # -----------------------------------------------------
    # SEND IMAGE TO AI MODEL
    # -----------------------------------------------------

    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )


    # -----------------------------------------------------
    # RUN AI MODEL
    # -----------------------------------------------------

    interpreter.invoke()


    # -----------------------------------------------------
    # GET AI OUTPUT
    # -----------------------------------------------------

    predictions = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]


    # -----------------------------------------------------
    # GET TOP 5 PREDICTIONS
    # -----------------------------------------------------

    top_indices = np.argsort(
        predictions
    )[::-1][:5]


    results = []


    for index in top_indices:

        if index < len(CLASS_NAMES):

            results.append(
                {
                    "name":
                        CLASS_NAMES[index],

                    "confidence":
                        float(
                            predictions[index]
                        )
                }
            )


    return results


# =========================================================
# CLEAR ADVICE FUNCTION
# =========================================================

def show_advice(
    prediction,
    confidence
):


    # =====================================================
    # LOW CONFIDENCE
    # =====================================================

    if confidence < 0.40:

        st.error(
            "⚠️ LOW CONFIDENCE"
        )

        st.write(
            "The AI is not confident enough "
            "to provide a reliable diagnosis."
        )

        st.write(
            "### 💡 What you should do:"
        )

        st.write(
            "📸 Take another clear close-up photograph."
        )

        st.write(
            "☀️ Use good natural lighting."
        )

        st.write(
            "🔍 Make sure the leaf is clearly visible."
        )

        st.write(
            "📱 Avoid blurry images."
        )

        st.write(
            "🍃 Photograph one leaf at a time."
        )

        st.write(
            "🔄 Try photographing both sides of the leaf."
        )

        st.write(
            "👨‍🌾 Consult an agricultural expert "
            "for confirmation."
        )

        st.warning(
            "⚠️ Do not apply pesticides or chemicals "
            "based only on a low-confidence AI prediction."
        )

        return


    # =====================================================
    # HEALTHY PLANT
    # =====================================================

    if "Healthy" in prediction:

        st.success(
            "🌿 THE PLANT APPEARS HEALTHY"
        )

        st.write(
            "### 💡 What you should do:"
        )

        st.write(
            "💧 Provide the plant with appropriate water."
        )

        st.write(
            "🌱 Provide appropriate nutrients."
        )

        st.write(
            "🧹 Keep the growing area clean."
        )

        st.write(
            "👀 Monitor the plant regularly."
        )

        st.write(
            "🍂 Remove dead or damaged plant material."
        )

        st.write(
            "🔍 Check regularly for new symptoms."
        )

        return


    # =====================================================
    # APPLE SCAB
    # =====================================================

    if "Apple - Apple Scab" in prediction:

        st.warning(
            "🍎 POSSIBLE APPLE SCAB"
        )

        st.write(
            "### 💡 What you should do:"
        )

        st.write(
            "🔍 Inspect other leaves and nearby apple trees."
        )

        st.write(
            "🍂 Remove severely affected plant material "
            "where appropriate."
        )

        st.write(
            "🧹 Keep fallen leaves and infected material "
            "away from the plant."
        )

        st.write(
            "🌬️ Improve air circulation around the tree."
        )

        st.write(
            "💧 Avoid unnecessary wetting of the leaves."
        )

        st.write(
            "👀 Monitor new leaves for additional symptoms."
        )

        st.write(
            "👨‍🌾 Consult an agricultural expert "
            "for confirmation."
        )

        st.error(
            "⚠️ Confirm the diagnosis before applying "
            "any chemical treatment."
        )

        return


    # =====================================================
    # APPLE BLACK ROT
    # =====================================================

    if "Apple - Black Rot" in prediction:

        st.warning(
            "🍎 POSSIBLE APPLE BLACK ROT"
        )

        st.write(
            "### 💡 What you should do:"
        )

        st.write(
            "🔍 Inspect the tree and nearby plants."
        )

        st.write(
            "🍂 Remove severely affected plant material "
            "where appropriate."
        )

        st.write(
            "🧹 Keep the growing area clean."
        )

        st.write(
            "🍎 Remove fallen infected fruit or leaves."
        )

        st.write(
            "👀 Monitor nearby plants for similar symptoms."
        )

        st.write(
            "👨‍🌾 Consult an agricultural expert "
            "for confirmation."
        )

        return


    # =====================================================
    # APPLE CEDAR APPLE RUST
    # =====================================================

    if "Apple - Cedar Apple Rust" in prediction:

        st.warning(
            "🍎 POSSIBLE APPLE CEDAR APPLE RUST"
        )

        st.write(
            "### 💡 What you should do:"
        )

        st.write(
            "🔍 Inspect other leaves for similar symptoms."
        )

        st.write(
            "👀 Monitor the tree regularly."
        )

        st.write(
            "🧹 Keep the growing area clean."
        )

        st.write(
            "🌬️ Improve air circulation around the tree."
        )

        st.write(
            "🍂 Remove severely affected leaves "
            "where appropriate."
        )

        st.write(
            "👨‍🌾 Consult an agricultural expert "
            "for confirmation and management advice."
        )

        return


    # =====================================================
    # TOMATO LATE BLIGHT
    # =====================================================

    if "Tomato - Late Blight" in prediction:

        st.warning(
            "🍅 POSSIBLE TOMATO LATE BLIGHT"
        )

        st.write(
            "### 💡 What you should do:"
        )

        st.write(
            "🔍 Inspect other tomato and potato plants nearby."
        )

        st.write(
            "👀 Monitor the crop for spreading symptoms."
        )

        st.write(
            "🌬️ Improve air circulation around plants."
        )

        st.write(
            "💧 Avoid unnecessary wetting of leaves."
        )

        st.write(
            "🍂 Remove severely affected material "
            "where appropriate."
        )

        st.write(
            "👨‍🌾 Seek agricultural advice for confirmation."
        )

        st.error(
            "⚠️ Confirm the diagnosis before beginning treatment."
        )

        return


    # =====================================================
    # TOMATO EARLY BLIGHT
    # =====================================================

    if "Tomato - Early Blight" in prediction:

        st.warning(
            "🍅 POSSIBLE TOMATO EARLY BLIGHT"
        )

        st.write(
            "### 💡 What you should do:"
        )

        st.write(
            "🔍 Inspect other tomato plants."
        )

        st.write(
            "👀 Monitor whether symptoms are spreading."
        )

        st.write(
            "🍂 Remove severely affected leaves "
            "where appropriate."
        )

        st.write(
            "🧹 Keep the area around plants clean."
        )

        st.write(
            "🌬️ Improve air circulation."
        )

        st.write(
            "👨‍🌾 Consult an agricultural expert "
            "for confirmation."
        )

        return


    # =====================================================
    # TOMATO LEAF MOLD
    # =====================================================

    if "Tomato - Leaf Mold" in prediction:

        st.warning(
            "🍅 POSSIBLE TOMATO LEAF MOLD"
        )

        st.write(
            "### 💡 What you should do:"
        )

        st.write(
            "🔍 Inspect other tomato plants."
        )

        st.write(
            "🌬️ Improve air circulation around plants."
        )

        st.write(
            "💧 Avoid unnecessary moisture on leaves."
        )

        st.write(
            "🍂 Remove severely affected leaves "
            "where appropriate."
        )

        st.write(
            "👀 Monitor the crop for spreading symptoms."
        )

        st.write(
            "👨‍🌾 Consult an agricultural expert "
            "for confirmation."
        )

        return


    # =====================================================
    # GRAPE DISEASES
    # =====================================================

    if "Grape -" in prediction:

        st.warning(
            "🍇 POSSIBLE GRAPE LEAF CONDITION"
        )

        st.write(
            "### 💡 What you should do:"
        )

        st.write(
            "🔍 Inspect other grape plants."
        )

        st.write(
            "👀 Monitor the crop for spreading symptoms."
        )

        st.write(
            "🍂 Remove severely affected material "
            "where appropriate."
        )

        st.write(
            "🧹 Keep the growing area clean."
        )

        st.write(
            "🌬️ Improve air circulation."
        )

        st.write(
            "👨‍🌾 Consult an agricultural professional "
            "for confirmation."
        )

        return


    # =====================================================
    # CORN DISEASES
    # =====================================================

    if "Corn -" in prediction:

        st.warning(
            "🌽 POSSIBLE CORN LEAF CONDITION"
        )

        st.write(
            "### 💡 What you should do:"
        )

        st.write(
            "🔍 Inspect other plants in the field."
        )

        st.write(
            "👀 Monitor whether symptoms are spreading."
        )

        st.write(
            "🧹 Keep the field clean."
        )

        st.write(
            "🔎 Check nearby plants for similar symptoms."
        )

        st.write(
            "👨‍🌾 Consult an agricultural extension officer."
        )

        return


    # =====================================================
    # POTATO DISEASES
    # =====================================================

    if "Potato -" in prediction:

        st.warning(
            "🥔 POSSIBLE POTATO LEAF CONDITION"
        )

        st.write(
            "### 💡 What you should do:"
        )

        st.write(
            "🔍 Inspect nearby potato plants."
        )

        st.write(
            "👀 Monitor the crop for spreading symptoms."
        )

        st.write(
            "🧹 Maintain good field hygiene."
        )

        st.write(
            "🍂 Remove severely affected material "
            "where appropriate."
        )

        st.write(
            "👨‍🌾 Consult an agricultural professional."
        )

        return


    # =====================================================
    # GENERAL DISEASE
    # =====================================================

    st.warning(
        "🌱 POSSIBLE PLANT CONDITION"
    )

    st.write(
        f"**AI Prediction:** {prediction}"
    )

    st.write(
        "### 💡 What you should do:"
    )

    st.write(
        "🔍 Inspect the plant carefully."
    )

    st.write(
        "👀 Check nearby plants for similar symptoms."
    )

    st.write(
        "📈 Monitor whether the condition is spreading."
    )

    st.write(
        "🧹 Keep the growing area clean."
    )

    st.write(
        "📸 Take another clear image if necessary."
    )

    st.write(
        "👨‍🌾 Consult an agricultural expert "
        "for confirmation."
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


# =========================================================
# UPLOAD IMAGE
# =========================================================

if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a clear plant leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


# =========================================================
# CAMERA
# =========================================================

else:

    uploaded_file = st.camera_input(
        "Take a clear picture of the leaf"
    )


# =========================================================
# IF IMAGE IS AVAILABLE
# =========================================================

if uploaded_file is not None:


    # Open image
    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # Display image
    st.image(
        image,
        caption="Uploaded Plant Leaf",
        use_container_width=True
    )


    st.divider()


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    if st.button(
        "🔍 ANALYZE LEAF",
        use_container_width=True
    ):


        try:


            # -------------------------------------------------
            # RUN AI
            # -------------------------------------------------

            with st.spinner(
                "🤖 AI is analyzing the leaf..."
            ):

                results = predict_leaf(
                    image
                )


            # -------------------------------------------------
            # BEST PREDICTION
            # -------------------------------------------------

            best_result = results[0]


            prediction = (
                best_result["name"]
            )


            confidence = (
                best_result["confidence"]
            )


            # -------------------------------------------------
            # TIME
            # -------------------------------------------------

            analysis_time = (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )


            # =================================================
            # RESULT
            # =================================================

            st.success(
                "✅ Analysis Complete!"
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


            # Confidence progress
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
            # CONFIDENCE STATUS
            # =================================================

            st.divider()


            if confidence < 0.40:


                st.error(
                    "⚠️ LOW CONFIDENCE"
                )


                st.write(
                    "The AI is uncertain about this result."
                )


            elif confidence < 0.70:


                st.warning(
                    "⚠️ MODERATE CONFIDENCE"
                )


                st.write(
                    "The AI has some uncertainty "
                    "about this result."
                )


            else:


                st.success(
                    "✅ HIGHER CONFIDENCE"
                )


                st.write(
                    "The AI has stronger confidence "
                    "in this prediction."
                )


            # =================================================
            # TOP 5 PREDICTIONS
            # =================================================

            st.divider()


            st.subheader(
                "🔎 Top 5 AI Predictions"
            )


            for number, result in enumerate(
                results,
                start=1
            ):


                st.write(
                    f"**{number}. "
                    f"{result['name']}** — "
                    f"{result['confidence'] * 100:.2f}%"
                )


            # =================================================
            # CLEAR ADVICE
            # =================================================

            st.divider()


            st.subheader(
                "💡 Clear Advice"
            )


            show_advice(
                prediction,
                confidence
            )


            # =================================================
            # DOWNLOAD REPORT
            # =================================================

            st.divider()


            st.subheader(
                "📄 Download Analysis Report"
            )


            report = f"""
SMART PLANT DISEASE DETECTOR
========================================

ANALYSIS TIME
----------------------------------------
{analysis_time}


AI PREDICTION
----------------------------------------
Prediction: {prediction}

Confidence:
{confidence * 100:.2f}%


TOP 5 PREDICTIONS
----------------------------------------
"""


            for number, result in enumerate(
                results,
                start=1
            ):


                report += (
                    f"{number}. "
                    f"{result['name']} - "
                    f"{result['confidence'] * 100:.2f}%\n"
                )


            report += """

IMPORTANT NOTICE
----------------------------------------

This AI system provides a supporting prediction.
It does not replace professional agricultural
diagnosis.

Confirm the diagnosis with a qualified
agricultural professional before making
important crop management decisions.


SMART PLANT DISEASE DETECTOR
AI FOR SMART AGRICULTURE
========================================
"""


            st.download_button(
                label="📥 Download Analysis Report",
                data=report,
                file_name=(
                    "plant_disease_analysis_report.txt"
                ),
                mime="text/plain",
                use_container_width=True
            )


        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except Exception as error:


            st.error(
                "❌ An error occurred while analyzing "
                "the image."
            )


            st.write(
                str(error)
            )


# =========================================================
# NO IMAGE
# =========================================================

else:


    st.info(
        "📷 Upload or capture a clear plant leaf image "
        "to begin analysis."
    )


# =========================================================
# ABOUT PROJECT
# =========================================================

st.divider()


st.subheader(
    "📖 About the Project"
)


st.write(
    "The Smart Plant Disease Detector is an "
    "Artificial Intelligence project designed "
    "to support smart agriculture."
)


st.write(
    "The system analyzes images of plant leaves "
    "and provides possible disease predictions, "
    "confidence scores and clear general advice."
)


st.write(
    "The system is intended as a supporting tool "
    "and should not replace professional agricultural "
    "diagnosis."
)


# =========================================================
# FOOTER
# =========================================================

st.divider()


st.caption(
    "🌿 Smart Plant Disease Detector | "
    "AI for Smart Agriculture"
)
