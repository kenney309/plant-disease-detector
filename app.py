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
# FAINT PROFESSIONAL GREEN DESIGN
# =========================================================

st.markdown(
    """
    <style>

    /* Very faint green background */
    .stApp {
        background:
        linear-gradient(
            rgba(245, 250, 245, 0.98),
            rgba(235, 248, 238, 0.98)
        );
    }

    /* Main title */
    h1 {
        color: #4F7F52;
        text-align: center;
        font-size: 42px;
        font-weight: bold;
    }

    /* Section titles */
    h2, h3 {
        color: #5F8F63;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        background-color: #6FA574;
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background-color: #5F8F63;
        color: white;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.65);
        border-radius: 15px;
        padding: 10px;
    }

    /* Result container */
    .result-box {
        padding: 20px;
        border-radius: 15px;
        background-color: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(95, 143, 99, 0.2);
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.title("🌿 Smart Plant Disease Detector")

st.markdown(
    """
    <div style="
        text-align:center;
        padding:15px;
        border-radius:15px;
        background-color:rgba(255,255,255,0.70);
        margin-bottom:20px;
    ">

    <h3>🤖 AI for Smart Agriculture</h3>

    <p>
    Upload or capture a clear image of a plant leaf.
    The AI will analyze the image and provide a possible
    disease prediction, confidence score and recommendation.
    </p>

    </div>
    """,
    unsafe_allow_html=True
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
def get_model():

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

    inputs = interpreter.get_input_details()

    outputs = interpreter.get_output_details()

    return (
        interpreter,
        inputs,
        outputs
    )


# =========================================================
# AI PREDICTION FUNCTION
# =========================================================

def analyze_leaf(image):

    (
        interpreter,
        inputs,
        outputs
    ) = get_model()

    input_shape = inputs[0]["shape"]

    height = input_shape[1]

    width = input_shape[2]

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

    prediction = interpreter.get_tensor(
        outputs[0]["index"]
    )[0]

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
# PLANT SELECTION
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
# IMAGE INPUT METHOD
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
# IMAGE ANALYSIS
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Plant Leaf",
        use_container_width=True
    )


    # =====================================================
    # MODEL LIMITATION WARNING
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

            with st.spinner(
                "🤖 AI is analyzing the leaf..."
            ):

                results = analyze_leaf(
                    image
                )


            # =================================================
            # MAIN RESULT
            # =================================================

            prediction = results[0][0]

            confidence = results[0][1]

            analysis_time = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )


            st.success(
                "✅ Analysis Complete!"
            )


            st.divider()

            st.subheader(
                "🌿 AI Prediction"
            )


            st.markdown(
                f"""
                <div class="result-box">

                <b>Plant Selected:</b> {plant}<br><br>

                <b>AI Prediction:</b> {prediction}<br><br>

                <b>Confidence:</b>
                {confidence * 100:.2f}%<br><br>

                <b>Analysis Time:</b>
                {analysis_time}

                </div>
                """,
                unsafe_allow_html=True
            )


            st.write(
                "### 📊 Confidence Level"
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
            # CONFIDENCE MESSAGE
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
                    this prediction. Consider verifying
                    the result with an agricultural expert.
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
                file_name="plant_disease_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )


        except Exception as error:

            st.error(
                "❌ An error occurred while analyzing the image."
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
