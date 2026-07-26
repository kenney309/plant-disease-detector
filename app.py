import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os
from datetime import datetime

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Smart Plant Disease Detector")

st.write(
    "Upload or capture a clear plant leaf image. "
    "The AI will analyze the image and provide a possible "
    "prediction, confidence score, and general guidance."
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
# DISEASE INFORMATION
# =========================================================

DISEASE_INFO = {

    "Apple - Apple Scab": {
        "symptoms": "Dark or olive-colored spots may appear on leaves and fruit.",
        "treatment": "Remove affected plant material and follow appropriate disease-management guidance.",
        "prevention": "Improve air circulation and remove fallen infected leaves."
    },

    "Grape - Black Rot": {
        "symptoms": "Dark spots and lesions may develop on leaves and fruit.",
        "treatment": "Remove infected material and follow suitable disease-management practices.",
        "prevention": "Maintain good vineyard sanitation and air circulation."
    },

    "Grape - Esca (Black Measles)": {
        "symptoms": "Leaves may develop irregular discoloration and the plant may weaken.",
        "treatment": "Remove severely affected plant parts and seek expert agricultural advice.",
        "prevention": "Use healthy planting material and maintain good plant hygiene."
    },

    "Grape - Leaf Blight": {
        "symptoms": "Brown or dark lesions can appear on leaves.",
        "treatment": "Remove infected leaves and use suitable disease-management methods.",
        "prevention": "Avoid excessive leaf wetness and improve air circulation."
    },

    "Grape - Healthy": {
        "symptoms": "No major disease symptoms were detected.",
        "treatment": "Continue normal plant care and monitoring.",
        "prevention": "Maintain good air circulation and monitor regularly."
    },

    "Tomato - Early Blight": {
        "symptoms": "Brown spots with ring-like patterns may develop on older leaves.",
        "treatment": "Remove affected leaves and follow appropriate disease-management guidance.",
        "prevention": "Practice crop rotation and avoid unnecessary leaf wetness."
    },

    "Tomato - Late Blight": {
        "symptoms": "Dark irregular lesions can develop rapidly on leaves and stems.",
        "treatment": "Remove severely infected material and seek agricultural advice.",
        "prevention": "Improve air circulation and avoid prolonged leaf wetness."
    },

    "Tomato - Healthy": {
        "symptoms": "No major disease symptoms were detected.",
        "treatment": "Continue normal plant care and monitor the plant regularly.",
        "prevention": "Maintain good sanitation, nutrition, and watering practices."
    },

    "Apple - Healthy": {
        "symptoms": "No major disease symptoms were detected.",
        "treatment": "Continue normal plant care.",
        "prevention": "Monitor the plant regularly and maintain good sanitation."
    }
}

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

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

                file.write(response.content)

    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    return (
        interpreter,
        interpreter.get_input_details(),
        interpreter.get_output_details()
    )

# =========================================================
# PREDICTION
# =========================================================

def predict_image(image):

    interpreter, inputs, outputs = load_model()

    input_shape = inputs[0]["shape"]

    height = input_shape[1]
    width = input_shape[2]

    image = image.resize(
        (width, height)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    interpreter.set_tensor(
        inputs[0]["index"],
        image_array
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        outputs[0]["index"]
    )[0]

    top_indices = np.argsort(
        prediction
    )[::-1][:3]

    results = []

    for index in top_indices:

        name = (
            CLASS_NAMES[index]
            if index < len(CLASS_NAMES)
            else "Unknown"
        )

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
# SIDEBAR
# =========================================================

st.sidebar.title("🌱 Plant Information")

plant = st.sidebar.selectbox(
    "Select the plant:",
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

st.subheader("📷 Upload or Capture a Leaf")

input_method = st.radio(
    "Choose image source:",
    [
        "Upload Image",
        "Use Camera"
    ],
    horizontal=True
)

uploaded_file = None

if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
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

    st.image(
        image,
        caption=f"{plant} Leaf Image",
        use_container_width=True
    )

    if plant in [
        "Guava",
        "Mango",
        "Banana"
    ]:

        st.warning(
            f"⚠️ The current AI model was not specifically "
            f"trained to recognize {plant}. "
            f"The result may therefore be inaccurate."
        )

    if st.button(
        "🔍 ANALYZE LEAF",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "🤖 AI is analyzing the leaf..."
            ):

                results = predict_image(
                    image
                )

            best_name = results[0][0]

            best_confidence = results[0][1]

            current_time = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            st.success(
                "Analysis complete!"
            )

            # =================================================
            # RESULT
            # =================================================

            st.subheader(
                "🌿 Analysis Result"
            )

            st.write(
                f"**Plant Selected:** {plant}"
            )

            st.write(
                f"**AI Prediction:** {best_name}"
            )

            st.write(
                f"**Confidence:** "
                f"{best_confidence * 100:.2f}%"
            )

            st.write(
                f"**Date and Time:** {current_time}"
            )

            st.progress(
                min(
                    max(
                        best_confidence,
                        0.0
                    ),
                    1.0
                )
            )

            # =================================================
            # CONFIDENCE WARNING
            # =================================================

            if best_confidence < 0.40:

                st.error(
                    "⚠️ LOW CONFIDENCE: "
                    "The AI is not confident about this result. "
                    "The plant may not be supported by the model."
                )

            elif best_confidence < 0.70:

                st.warning(
                    "⚠️ MODERATE CONFIDENCE: "
                    "Consider verifying the result."
                )

            else:

                st.success(
                    "✅ The AI has a higher confidence "
                    "in this prediction."
                )

            # =================================================
            # TOP 3
            # =================================================

            st.subheader(
                "🔎 Top 3 Predictions"
            )

            for number, (
                name,
                score
            ) in enumerate(
                results,
                start=1
            ):

                st.write(
                    f"**{number}. {name}** — "
                    f"{score * 100:.2f}%"
                )

            # =================================================
            # INFORMATION
            # =================================================

            if best_name in DISEASE_INFO:

                info = DISEASE_INFO[
                    best_name
                ]

                st.divider()

                st.subheader(
                    "📚 Disease Information"
                )

                st.write(
                    f"**Symptoms:** "
                    f"{info['symptoms']}"
                )

                st.write(
                    f"**General Management:** "
                    f"{info['treatment']}"
                )

                st.write(
                    f"**Prevention:** "
                    f"{info['prevention']}"
                )

                symptoms = info["symptoms"]
                treatment = info["treatment"]
                prevention = info["prevention"]

            else:

                symptoms = (
                    "Detailed symptom information "
                    "is not currently available."
                )

                treatment = (
                    "Seek advice from a qualified "
                    "agricultural professional."
                )

                prevention = (
                    "Monitor the plant regularly "
                    "and maintain good plant hygiene."
                )

                st.info(
                    "Detailed information for this "
                    "prediction is not yet available."
                )

            # =================================================
            # DOWNLOADABLE REPORT
            # =================================================

            st.divider()

            st.subheader(
                "📄 Download Analysis Report"
            )

            report = f"""
SMART PLANT DISEASE DETECTOR
================================

PLANT INFORMATION
-----------------
Selected Plant: {plant}

ANALYSIS INFORMATION
--------------------
Date and Time: {current_time}

AI PREDICTION
-------------
Prediction: {best_name}
Confidence: {best_confidence * 100:.2f}%

TOP 3 AI PREDICTIONS
--------------------
1. {results[0][0]} - {results[0][1] * 100:.2f}%

2. {results[1][0]} - {results[1][1] * 100:.2f}%

3. {results[2][0]} - {results[2][1] * 100:.2f}%

DISEASE INFORMATION
-------------------
Symptoms:
{symptoms}

General Management:
{treatment}

Prevention:
{prevention}

IMPORTANT NOTICE
----------------
This AI result is for educational and supporting
purposes only. It should not replace professional
agricultural diagnosis. If the result has low confidence,
consult a qualified agricultural expert.

================================
Smart Plant Disease Detector
AI for Smart Agriculture
"""

            st.download_button(
                label="📥 Download Analysis Report",
                data=report,
                file_name="plant_disease_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        except Exception as e:

            st.error(
                "❌ An error occurred during analysis."
            )

            st.write(
                str(e)
            )

else:

    st.info(
        "📷 Upload or capture a clear leaf image "
        "to begin analysis."
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
    The Smart Plant Disease Detector is an Artificial
    Intelligence project designed to support smart agriculture.

    The system analyzes plant leaf images and provides
    possible disease predictions, confidence scores,
    and general agricultural guidance.

    The system is intended as a supporting tool and should
    not replace professional agricultural diagnosis.
    """
)

st.caption(
    "🌿 Smart Plant Disease Detector | AI for Smart Agriculture"
)
