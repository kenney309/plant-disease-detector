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
    "disease prediction, confidence score and recommendation."
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
# IMPORTANT: KEEP THE ORDER EXACTLY AS THE MODEL OUTPUT
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
def get_model():

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

def analyze_leaf(image):

    (
        interpreter,
        input_details,
        output_details
    ) = get_model()

    # Get model input size
    input_shape = input_details[0]["shape"]

    height = input_shape[1]

    width = input_shape[2]

    # Resize image
    image = image.resize(
        (
            width,
            height
        )
    )

    # Convert image to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize image
    image_array = (
        image_array / 255.0
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Send image to model
    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )

    # Run AI model
    interpreter.invoke()

    # Get prediction
    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    # Get top 3 predictions
    top_indices = np.argsort(
        prediction
    )[::-1][:3]

    results = []

    for index in top_indices:

        if index < len(CLASS_NAMES):

            name = CLASS_NAMES[index]

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
# SELECT PLANT
# =========================================================

st.subheader(
    "🌱 Select Your Plant"
)

plant = st.selectbox(
    "Choose the plant you are testing:",
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
# IMAGE SOURCE
# =========================================================

st.subheader(
    "📷 Choose Image Source"
)

input_method = st.radio(
    "How would you like to provide the leaf image?",
    [
        "Upload Image",
        "Use Camera"
    ],
    horizontal=True
)


uploaded_file = None


# Upload image
if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a clear plant leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


# Use camera
else:

    uploaded_file = st.camera_input(
        "Take a clear picture of the leaf"
    )


# =========================================================
# DISPLAY AND ANALYZE IMAGE
# =========================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Plant Leaf",
        use_container_width=True
    )


    # =====================================================
    # MODEL SUPPORT WARNING
    # =====================================================

    if plant in [
        "Guava",
        "Mango",
        "Banana"
    ]:

        st.warning(
            f"""
            ⚠️ Important Notice:

            The current AI model was not specifically
            trained to recognize {plant}.

            The prediction may therefore be inaccurate.
            """
        )


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    if st.button(
        "🔍 ANALYZE LEAF",
        use_container_width=True
    ):

        try:

            # Analyze image
            with st.spinner(
                "🤖 AI is analyzing the leaf..."
            ):

                results = analyze_leaf(
                    image
                )


            # Get best prediction
            prediction = results[0][0]

            confidence = results[0][1]


            # Get current time
            analysis_time = (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )


            st.success(
                "✅ Analysis Complete!"
            )


            st.divider()


            # =================================================
            # MAIN RESULT
            # =================================================

            st.subheader(
                "🌿 AI Prediction"
            )

            st.write(
                f"**Plant Selected:** {plant}"
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


            # Confidence progress bar
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
            # CONFIDENCE WARNING
            # =================================================

            if confidence < 0.40:

                st.error(
                    """
                    ⚠️ LOW CONFIDENCE

                    The AI is not confident about this
                    prediction. The result may be incorrect.

                    Try taking another clear photograph
                    with good lighting.
                    """
                )


            elif confidence < 0.70:

                st.warning(
                    """
                    ⚠️ MODERATE CONFIDENCE

                    The AI has some uncertainty about
                    this prediction.

                    Consider verifying the result with
                    an agricultural expert.
                    """
                )


            else:

                st.success(
                    """
                    ✅ HIGHER CONFIDENCE

                    The AI has a relatively higher
                    confidence in this prediction.
                    """
                )


            # =================================================
            # TOP 3 PREDICTIONS
            # =================================================

            st.divider()

            st.subheader(
                "🔎 Top 3 AI Predictions"
            )


            for number, result in enumerate(
                results,
                start=1
            ):

                name = result[0]

                score = result[1]

                st.write(
                    f"**{number}. {name}** — "
                    f"{score * 100:.2f}%"
                )


            # =================================================
            # RECOMMENDATIONS
            # =================================================

            st.divider()

            st.subheader(
                "💡 Recommendations"
            )


            # Healthy plant
            if "Healthy" in prediction:

                st.success(
                    """
                    🌿 The plant appears healthy
                    according to the AI prediction.

                    Recommended actions:

                    • Continue regular plant care.
                    • Monitor the plant regularly.
                    • Maintain good sanitation.
                    • Provide appropriate water and nutrients.
                    • Keep the growing area clean.
                    """
                )


            # Low confidence
            elif confidence < 0.40:

                st.info(
                    """
                    🌱 The AI result is uncertain.

                    Recommended actions:

                    • Take another clear image.
                    • Use good natural lighting.
                    • Make sure the leaf is clearly visible.
                    • Avoid blurry photographs.
                    • Consult an agricultural expert
                      for a confirmed diagnosis.
                    """
                )


            # Possible disease
            else:

                st.warning(
                    f"""
                    🌱 Possible condition detected:

                    {prediction}

                    General recommendations:

                    • Monitor the plant closely.
                    • Remove severely affected leaves
                      when appropriate.
                    • Keep the growing area clean.
                    • Improve air circulation.
                    • Avoid unnecessary leaf wetness.
                    • Check nearby plants for similar symptoms.
                    • Consult an agricultural expert for
                      confirmed diagnosis and treatment.
                    """
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
================================

PLANT INFORMATION
--------------------------------
Selected Plant: {plant}

ANALYSIS INFORMATION
--------------------------------
Date and Time: {analysis_time}

AI PREDICTION
--------------------------------
Prediction: {prediction}

Confidence:
{confidence * 100:.2f}%


TOP 3 PREDICTIONS
--------------------------------

1. {results[0][0]}
Confidence: {results[0][1] * 100:.2f}%

2. {results[1][0]}
Confidence: {results[1][1] * 100:.2f}%

3. {results[2][0]}
Confidence: {results[2][1] * 100:.2f}%


RECOMMENDATION
--------------------------------

The AI prediction is intended as a supporting
tool and should not replace professional
agricultural diagnosis.

If the confidence is low, take another clear
image or consult a qualified agricultural expert.


SMART PLANT DISEASE DETECTOR
AI FOR SMART AGRICULTURE
================================
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


        # =================================================
        # ERROR HANDLING
        # =================================================

        except Exception as error:

            st.error(
                "❌ An error occurred while analyzing "
                "the image."
            )

            st.write(
                str(error)
            )


# =========================================================
# NO IMAGE MESSAGE
# =========================================================

else:

    st.info(
        "📷 Upload or capture a clear plant leaf image "
        "to begin analysis."
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
    The Smart Plant Disease Detector is an
    Artificial Intelligence project designed
    to support smart agriculture.

    The system analyzes images of plant leaves
    and provides possible disease predictions,
    confidence scores and general recommendations.

    The system is designed as a supporting tool
    and should not replace professional agricultural
    diagnosis.
    """
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🌿 Smart Plant Disease Detector | "
    "AI for Smart Agriculture"
)
