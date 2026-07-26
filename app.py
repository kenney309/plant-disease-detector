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
    "Upload a clear image of a plant leaf. "
    "The AI will analyze the image and provide a "
    "possible plant health prediction."
)

st.info(
    "📸 For the best result, photograph one leaf clearly "
    "in good lighting. Avoid blurry images and busy backgrounds."
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
# MODEL CLASS NAMES
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
# DOWNLOAD AND LOAD MODEL
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

                file.write(
                    response.content
                )

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
# AI PREDICTION
# =========================================================

def predict_leaf(image):

    (
        interpreter,
        input_details,
        output_details
    ) = load_model()


    # Get the model's actual input dimensions
    input_shape = input_details[0]["shape"]

    height = int(input_shape[1])

    width = int(input_shape[2])


    # Resize image
    image = image.resize(
        (width, height)
    )


    # Convert image to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
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


    # Run model
    interpreter.invoke()


    # Get output
    predictions = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]


    # Get top 5 predictions
    top_indices = np.argsort(
        predictions
    )[::-1][:5]


    results = []


    for index in top_indices:

        if index < len(CLASS_NAMES):

            results.append(
                {
                    "name": CLASS_NAMES[index],
                    "confidence": float(
                        predictions[index]
                    )
                }
            )


    return results


# =========================================================
# CLEAR ADVICE FUNCTION
# =========================================================

def get_advice(prediction, confidence):

    # Low confidence
    if confidence < 0.40:

        return """
⚠️ THE AI IS NOT CONFIDENT ENOUGH

The prediction is uncertain and should not be treated
as a confirmed diagnosis.

What you should do:

• Take another clear close-up photograph.
• Use good natural lighting.
• Make sure the leaf is not blurry.
• Photograph one leaf at a time.
• Make sure the affected area is visible.
• Try photographing both sides of the leaf.
• Consult an agricultural expert for confirmation.

Do not apply pesticides or chemicals based only on
this low-confidence AI prediction.
"""


    # Healthy
    if "Healthy" in prediction:

        return """
🌿 THE LEAF APPEARS HEALTHY

The AI predicts that the leaf may be healthy.

What you should do:

• Continue regular plant care.
• Monitor the plant regularly.
• Provide appropriate water and nutrients.
• Keep the growing area clean.
• Remove dead or damaged plant material.
• Check regularly for new symptoms.

If you notice unusual spots, discoloration, or other
symptoms, take another clear image for analysis.
"""


    # Apple Scab
    if "Apple - Apple Scab" in prediction:

        return """
🍎 POSSIBLE APPLE SCAB

The AI has detected a possible Apple Scab condition.

What you should do:

• Inspect other leaves and nearby apple trees.
• Remove severely affected plant material where appropriate.
• Keep fallen leaves and infected material away from the plant.
• Improve air circulation around the tree.
• Avoid unnecessary wetting of leaves.
• Monitor new leaves for additional symptoms.
• Consult an agricultural expert for confirmation.

Confirm the diagnosis before applying any chemical treatment.
"""


    # Apple Black Rot
    if "Apple - Black Rot" in prediction:

        return """
🍎 POSSIBLE APPLE BLACK ROT

The AI has detected a possible Black Rot condition.

What you should do:

• Inspect the tree and nearby plants carefully.
• Remove severely affected material where appropriate.
• Keep the growing area clean.
• Remove fallen infected plant material.
• Monitor nearby plants for similar symptoms.
• Seek professional agricultural advice for confirmation.

Do not apply chemical treatment based only on an AI prediction.
"""


    # Tomato Late Blight
    if "Tomato - Late Blight" in prediction:

        return """
🍅 POSSIBLE TOMATO LATE BLIGHT

The AI has detected a possible Late Blight condition.

What you should do:

• Inspect other tomato and potato plants nearby.
• Monitor the crop closely for spreading symptoms.
• Improve air circulation around plants.
• Avoid unnecessary wetting of leaves.
• Remove severely affected material where appropriate.
• Seek agricultural advice promptly for confirmation.

Late Blight can spread quickly, so professional confirmation
is recommended.
"""


    # Tomato Early Blight
    if "Tomato - Early Blight" in prediction:

        return """
🍅 POSSIBLE TOMATO EARLY BLIGHT

The AI has detected a possible Early Blight condition.

What you should do:

• Inspect other tomato plants.
• Monitor whether symptoms are spreading.
• Remove severely affected leaves where appropriate.
• Keep the area around the plants clean.
• Improve air circulation.
• Consult an agricultural expert for confirmation.
"""


    # Grape diseases
    if "Grape -" in prediction:

        return """
🍇 POSSIBLE GRAPE LEAF CONDITION

The AI has detected a possible grape leaf condition.

What you should do:

• Inspect other grape plants.
• Monitor the crop for spreading symptoms.
• Remove severely affected plant material where appropriate.
• Keep the growing area clean.
• Improve air circulation.
• Consult an agricultural professional for confirmation.
"""


    # Corn diseases
    if "Corn -" in prediction:

        return """
🌽 POSSIBLE CORN LEAF CONDITION

The AI has detected a possible corn leaf condition.

What you should do:

• Inspect other plants in the field.
• Monitor whether symptoms are spreading.
• Keep the field clean.
• Check nearby plants for similar symptoms.
• Seek advice from an agricultural extension officer
  or qualified agricultural professional.
"""


    # Potato diseases
    if "Potato -" in prediction:

        return """
🥔 POSSIBLE POTATO LEAF CONDITION

The AI has detected a possible potato leaf condition.

What you should do:

• Inspect nearby potato plants.
• Monitor the crop for spreading symptoms.
• Maintain good field hygiene.
• Remove severely affected material where appropriate.
• Consult an agricultural professional for confirmation.
"""


    # General disease
    return f"""
🌱 POSSIBLE PLANT CONDITION

AI Prediction:
{prediction}

What you should do:

• Inspect the plant carefully.
• Check nearby plants for similar symptoms.
• Monitor whether the condition is spreading.
• Keep the growing area clean.
• Take another clear photo if necessary.
• Consult an agricultural expert for confirmation.

IMPORTANT:
The AI prediction is a supporting tool and should not
replace professional agricultural diagnosis.
"""


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

    uploaded_file = st.file_uploader(
        "Upload a clear plant leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

else:

    uploaded_file = st.camera_input(
        "Take a clear picture of the leaf"
    )


# =========================================================
# ANALYZE IMAGE
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.image(
        image,
        caption="Plant Leaf Image",
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


            # Best result
            best_result = results[0]

            prediction = (
                best_result["name"]
            )

            confidence = (
                best_result["confidence"]
            )


            analysis_time = (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )


            # =================================================
            # AI PREDICTION
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

            if confidence < 0.40:

                st.error(
                    "⚠️ LOW CONFIDENCE — "
                    "The AI is uncertain about this result."
                )

            elif confidence < 0.70:

                st.warning(
                    "⚠️ MODERATE CONFIDENCE — "
                    "Consider confirming the result."
                )

            else:

                st.success(
                    "✅ HIGHER CONFIDENCE — "
                    "The AI has stronger confidence in this result."
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


            advice = get_advice(
                prediction,
                confidence
            )


            st.info(
                advice
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


            report += f"""

CLEAR ADVICE
----------------------------------------
{advice}


IMPORTANT NOTICE
----------------------------------------

This AI system provides a supporting prediction
and does not replace professional agricultural
diagnosis.

The model may perform differently on real-world
field photographs because lighting, background,
leaf position, and image quality can affect results.

For important crop management decisions, consult
a qualified agricultural professional.


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


        except Exception as error:

            st.error(
                "❌ An error occurred while analyzing "
                "the image."
            )

            st.write(
                str(error)
            )


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
    """
    The Smart Plant Disease Detector is an
    Artificial Intelligence project designed
    to support smart agriculture.

    The system analyzes images of plant leaves
    and provides possible disease predictions,
    confidence scores and clear general advice.

    The system is intended as a supporting tool
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
