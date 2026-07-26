import streamlit as st
from PIL import Image, ImageStat
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
    page_icon="Plant",
    layout="centered"
)


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0


# =========================================================
# CUSTOM STYLE
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #cccccc;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .section-title {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">Smart Plant Disease Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI for Smart Agriculture</div>',
    unsafe_allow_html=True
)

st.write(
    "This system analyzes plant leaf images and provides "
    "possible disease predictions, confidence scores, "
    "general management advice and prevention information."
)

st.info(
    "For better results, use a clear close-up image of one "
    "plant leaf with good lighting."
)

st.divider()


# =========================================================
# LANGUAGE
# =========================================================

language = st.selectbox(
    "Choose Language",
    ["English", "Luganda"]
)

if language == "Luganda":

    st.info(
        "Olonda Olulimi: Luganda. "
        "Ebimu ku bikwata ku pulogulaamu bikyali mu Lungereza."
    )


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
# SUPPORTED PLANTS
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
# PLANT SELECTION
# =========================================================

st.subheader("Plant Selection")

plant = st.selectbox(
    "Select the plant you want to analyze",
    SUPPORTED_PLANTS
)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner(
            "Downloading the AI model for the first time..."
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

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    return (
        interpreter,
        input_details,
        output_details
    )


# =========================================================
# IMAGE QUALITY CHECK
# =========================================================

def check_image_quality(image):

    gray_image = image.convert("L")

    brightness = ImageStat.Stat(
        gray_image
    ).mean[0]

    width, height = image.size

    if width < 200 or height < 200:

        return (
            False,
            "Image resolution is too low. "
            "Please use a clearer image."
        )

    if brightness < 35:

        return (
            False,
            "Image is too dark. "
            "Please take the photo in better lighting."
        )

    if brightness > 245:

        return (
            False,
            "Image is too bright. "
            "Please reduce excessive lighting."
        )

    return (
        True,
        "Image quality appears suitable."
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

    input_shape = input_details[0]["shape"]

    height = int(
        input_shape[1]
    )

    width = int(
        input_shape[2]
    )

    image = image.resize(
        (width, height)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )

    interpreter.invoke()

    predictions = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    if (
        np.min(predictions) < 0
        or np.max(predictions) > 1
        or not np.isclose(
            np.sum(predictions),
            1.0,
            atol=0.05
        )
    ):

        exp_predictions = np.exp(
            predictions - np.max(predictions)
        )

        predictions = (
            exp_predictions
            / np.sum(exp_predictions)
        )

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
# DISEASE INFORMATION
# =========================================================

DISEASE_INFO = {

    "Apple - Apple Scab":
        "Apple scab is a fungal disease that can affect "
        "apple leaves and fruit. Symptoms may include "
        "dark or olive-colored spots.",

    "Apple - Black Rot":
        "Black rot is a fungal disease that can affect "
        "apple leaves, fruit and branches.",

    "Apple - Cedar Apple Rust":
        "Cedar apple rust is a fungal disease that can "
        "cause yellow or orange symptoms on leaves.",

    "Corn - Common Rust":
        "Common rust is a fungal disease of corn that "
        "can produce reddish-brown rust-like spots.",

    "Corn - Northern Leaf Blight":
        "Northern leaf blight can cause long gray or "
        "brown lesions on corn leaves.",

    "Tomato - Early Blight":
        "Early blight can cause dark spots and lesions "
        "on tomato leaves.",

    "Tomato - Late Blight":
        "Late blight is a serious disease that can "
        "affect tomato and potato crops.",

    "Potato - Early Blight":
        "Potato early blight can cause dark lesions "
        "on potato leaves.",

    "Potato - Late Blight":
        "Potato late blight can spread rapidly under "
        "favorable conditions.",

    "Grape - Black Rot":
        "Grape black rot is a fungal disease that can "
        "affect grape leaves and fruit.",

    "Grape - Esca (Black Measles)":
        "Esca is a complex grapevine disease associated "
        "with wood-inhabiting fungi.",

    "Grape - Leaf Blight":
        "Grape leaf blight can cause damage to grape "
        "leaves and affect plant growth."
}


# =========================================================
# OFFLINE PLANT HEALTH GUIDE
# =========================================================

OFFLINE_GUIDE = {

    "Apple": {
        "Common conditions":
            "Apple scab, black rot and cedar apple rust.",
        "General advice":
            "Inspect leaves regularly, keep the growing "
            "area clean and improve air circulation."
    },

    "Blueberry": {
        "Common conditions":
            "The current model mainly provides a healthy "
            "classification for blueberry.",
        "General advice":
            "Monitor plants regularly and inspect for "
            "unusual leaf or fruit symptoms."
    },

    "Cherry": {
        "Common conditions":
            "Powdery mildew and other plant health problems.",
        "General advice":
            "Maintain good air circulation and monitor "
            "new growth."
    },

    "Corn": {
        "Common conditions":
            "Common rust, northern leaf blight and "
            "cercospora leaf spot.",
        "General advice":
            "Inspect plants regularly and maintain good "
            "field hygiene."
    },

    "Grape": {
        "Common conditions":
            "Black rot, Esca and leaf blight.",
        "General advice":
            "Monitor leaves and fruit and maintain good "
            "air circulation."
    },

    "Orange": {
        "Common conditions":
            "The current model includes citrus greening.",
        "General advice":
            "Inspect trees regularly and seek expert advice "
            "when unusual symptoms appear."
    },

    "Peach": {
        "Common conditions":
            "Bacterial spot.",
        "General advice":
            "Monitor leaves and fruit and maintain good "
            "orchard hygiene."
    },

    "Pepper": {
        "Common conditions":
            "Bacterial spot.",
        "General advice":
            "Use healthy planting material and monitor "
            "plants regularly."
    },

    "Potato": {
        "Common conditions":
            "Early blight and late blight.",
        "General advice":
            "Inspect plants regularly and seek expert "
            "advice when symptoms appear."
    },

    "Raspberry": {
        "Common conditions":
            "The current model mainly provides a healthy "
            "classification for raspberry.",
        "General advice":
            "Monitor plants regularly for unusual symptoms."
    },

    "Soybean": {
        "Common conditions":
            "The current model mainly provides a healthy "
            "classification for soybean.",
        "General advice":
            "Inspect plants regularly and maintain good "
            "field hygiene."
    },

    "Squash": {
        "Common conditions":
            "Powdery mildew.",
        "General advice":
            "Maintain good air circulation and monitor "
            "leaves regularly."
    },

    "Strawberry": {
        "Common conditions":
            "Leaf scorch.",
        "General advice":
            "Monitor leaves and maintain good growing "
            "conditions."
    },

    "Tomato": {
        "Common conditions":
            "Bacterial spot, early blight, late blight, "
            "leaf mold, septoria leaf spot, spider mites, "
            "target spot and viral diseases.",
        "General advice":
            "Inspect plants regularly and seek professional "
            "confirmation before applying treatments."
    }
}


# =========================================================
# GENERAL MANAGEMENT ADVICE
# =========================================================

def show_advice():

    st.write(
        "Inspect the affected plant and nearby plants "
        "for similar symptoms."
    )

    st.write(
        "Remove severely affected plant material "
        "where appropriate."
    )

    st.write(
        "Keep the growing area clean."
    )

    st.write(
        "Improve air circulation around plants."
    )

    st.write(
        "Avoid unnecessary wetting of leaves."
    )

    st.write(
        "Monitor the plant regularly."
    )

    st.write(
        "Consult an agricultural expert for confirmation."
    )


# =========================================================
# PREVENTION
# =========================================================

def show_prevention():

    st.write(
        "Use healthy planting materials."
    )

    st.write(
        "Keep the field or garden clean."
    )

    st.write(
        "Inspect plants regularly."
    )

    st.write(
        "Maintain good spacing and air circulation."
    )

    st.write(
        "Avoid unnecessary moisture on leaves."
    )

    st.write(
        "Follow recommendations from agricultural experts."
    )


# =========================================================
# OFFLINE MODE
# =========================================================

st.divider()

st.subheader("Offline Plant Health Guide")

st.write(
    "This section provides basic plant health information "
    "without requiring an AI prediction."
)

offline_plant = st.selectbox(
    "Select a plant for offline information",
    SUPPORTED_PLANTS,
    key="offline_plant"
)

if offline_plant in OFFLINE_GUIDE:

    guide = OFFLINE_GUIDE[
        offline_plant
    ]

    st.write(
        "Plant: "
        + offline_plant
    )

    st.write(
        "Common conditions: "
        + guide["Common conditions"]
    )

    st.write(
        "General advice: "
        + guide["General advice"]
    )

    st.warning(
        "Offline information is general guidance and "
        "does not replace professional diagnosis."
    )


# =========================================================
# IMAGE INPUT
# =========================================================

st.divider()

st.subheader(
    "Plant Leaf Analysis"
)

input_method = st.radio(
    "Choose how to provide the leaf image",
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
        "Take a clear photograph of the leaf"
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
        caption="Selected Plant Leaf",
        use_container_width=True
    )

    quality_ok, quality_message = (
        check_image_quality(image)
    )

    if quality_ok:

        st.success(
            quality_message
        )

    else:

        st.warning(
            quality_message
        )


    st.divider()

    analyze_button = st.button(
        "ANALYZE LEAF",
        use_container_width=True
    )


    if analyze_button:

        if not quality_ok:

            st.warning(
                "Please provide a clearer image before "
                "continuing with the analysis."
            )

        else:

            try:

                with st.spinner(
                    "AI is analyzing the plant leaf..."
                ):

                    results = predict_leaf(
                        image
                    )


                if not results:

                    st.error(
                        "No prediction was produced."
                    )

                    st.stop()


                prediction = (
                    results[0]["name"]
                )

                confidence = (
                    results[0]["confidence"]
                )

                analysis_time = (
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )


                st.session_state.last_prediction = (
                    prediction
                )

                st.session_state.last_confidence = (
                    confidence
                )


                # =========================================
                # MAIN RESULT
                # =========================================

                st.divider()

                st.subheader(
                    "AI Prediction"
                )

                st.write(
                    "Plant Selected: "
                    + plant
                )

                st.write(
                    "Prediction: "
                    + prediction
                )

                st.write(
                    "Confidence: "
                    + f"{confidence * 100:.2f}%"
                )

                st.write(
                    "Analysis Time: "
                    + analysis_time
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


                # =========================================
                # PLANT MATCH
                # =========================================

                predicted_plant = (
                    prediction.split(
                        " - "
                    )[0]
                )


                if predicted_plant != plant:

                    st.warning(
                        "The selected plant is "
                        + plant
                        + ", but the AI predicted a class "
                        + "belonging to "
                        + predicted_plant
                        + ". The result may be unreliable."
                    )


                # =========================================
                # CONFIDENCE
                # =========================================

                if confidence < 0.40:

                    st.error(
                        "LOW CONFIDENCE"
                    )

                    st.write(
                        "The AI is not confident about this "
                        "prediction. The result may be incorrect."
                    )

                    st.write(
                        "Take another clear photograph "
                        "with good lighting."
                    )

                    st.write(
                        "Make sure the leaf is clearly visible."
                    )

                    st.write(
                        "Consult an agricultural expert "
                        "for confirmation."
                    )

                elif confidence < 0.70:

                    st.warning(
                        "MODERATE CONFIDENCE"
                    )

                    st.write(
                        "The AI has some uncertainty about "
                        "this result."
                    )

                else:

                    st.success(
                        "HIGHER CONFIDENCE"
                    )


                # =========================================
                # TOP 5
                # =========================================

                st.divider()

                st.subheader(
                    "Top 5 AI Predictions"
                )

                chart_data = {}

                for number, result in enumerate(
                    results,
                    start=1
                ):

                    name = result["name"]

                    score = result["confidence"]

                    st.write(
                        str(number)
                        + ". "
                        + name
                        + " - "
                        + f"{score * 100:.2f}%"
                    )

                    chart_data[name] = score


                st.bar_chart(
                    chart_data
                )


                # =========================================
                # DISEASE INFORMATION
                # =========================================

                st.divider()

                st.subheader(
                    "Disease Information"
                )


                if "Healthy" in prediction:

                    st.success(
                        "The AI prediction indicates that "
                        "the plant may be healthy."
                    )

                    st.write(
                        "Continue monitoring the plant regularly "
                        "for new symptoms."
                    )

                elif prediction in DISEASE_INFO:

                    st.write(
                        DISEASE_INFO[prediction]
                    )

                else:

                    st.write(
                        "The AI has identified a possible "
                        "plant health condition. Further "
                        "professional assessment is needed "
                        "to confirm the diagnosis."
                    )


                # =========================================
                # MANAGEMENT
                # =========================================

                st.divider()

                st.subheader(
                    "Recommended Actions"
                )

                show_advice()


                # =========================================
                # PREVENTION
                # =========================================

                st.divider()

                st.subheader(
                    "Prevention Tips"
                )

                show_prevention()


                # =========================================
                # HEALTH STATUS
                # =========================================

                st.divider()

                st.subheader(
                    "Plant Health Status"
                )


                if "Healthy" in prediction:

                    health_status = "Healthy"

                elif confidence >= 0.70:

                    health_status = "High Risk"

                elif confidence >= 0.40:

                    health_status = "Needs Attention"

                else:

                    health_status = "Uncertain"


                st.write(
                    "Current Plant Health Status: "
                    + health_status
                )


                # =========================================
                # SAVE HISTORY
                # =========================================

                st.session_state.history.append(
                    {
                        "Time": analysis_time,
                        "Plant": plant,
                        "Prediction": prediction,
                        "Confidence":
                            f"{confidence * 100:.2f}%",
                        "Status": health_status
                    }
                )


            except Exception as error:

                st.error(
                    "An error occurred while analyzing "
                    "the image."
                )

                st.code(
                    str(error)
                )


else:

    st.info(
        "Upload or capture a clear plant leaf image "
        "to begin analysis."
    )


# =========================================================
# PLANT DOCTOR
# =========================================================

st.divider()

st.subheader(
    "Ask the Plant Doctor"
)

st.write(
    "Ask a question about plant diseases, prevention "
    "or general management."
)

question = st.text_input(
    "Type your question here"
)


if st.button(
    "GET PLANT ADVICE",
    use_container_width=True
):

    if question.strip() == "":

        st.warning(
            "Please type a question first."
        )

    else:

        question_lower = (
            question.lower()
        )

        st.success(
            "Plant Doctor Advice"
        )


        if (
            "what is" in question_lower
            or "meaning" in question_lower
        ):

            st.write(
                "The AI prediction represents a possible "
                "plant health condition identified from "
                "the uploaded image."
            )

            st.write(
                "For accurate diagnosis, compare symptoms "
                "with reliable agricultural information "
                "and consult an agricultural professional."
            )


        elif (
            "treat" in question_lower
            or "treatment" in question_lower
            or "cure" in question_lower
        ):

            st.write(
                "Confirm the diagnosis before applying "
                "any treatment."
            )

            st.write(
                "Inspect nearby plants."
            )

            st.write(
                "Remove severely affected plant material "
                "where appropriate."
            )

            st.write(
                "Improve air circulation."
            )

            st.write(
                "Avoid unnecessary wetting of leaves."
            )

            st.write(
                "Consult an agricultural expert."
            )


        elif (
            "prevent" in question_lower
            or "avoid" in question_lower
        ):

            show_prevention()


        elif (
            "spread" in question_lower
            or "spreading" in question_lower
        ):

            st.write(
                "Some plant diseases can spread through "
                "water, insects, contaminated tools, "
                "infected plant material or environmental "
                "conditions."
            )

            st.write(
                "Inspect nearby plants and monitor them "
                "regularly."
            )


        elif (
            "healthy" in question_lower
        ):

            st.write(
                "Continue monitoring the plant regularly."
            )

            st.write(
                "Provide appropriate water and nutrients."
            )

            st.write(
                "Check regularly for new symptoms."
            )


        else:

            if st.session_state.last_prediction:

                st.write(
                    "Latest AI prediction: "
                    + st.session_state.last_prediction
                )

            st.write(
                "Inspect the plant carefully, monitor "
                "nearby plants and consult an agricultural "
                "expert if symptoms continue."
            )

            st.info(
                "You can ask questions such as "
                "What is this disease? "
                "How can I prevent it? "
                "Can it spread? "
                "or How should I manage it?"
            )


# =========================================================
# ANALYSIS HISTORY
# =========================================================

st.divider()

st.subheader(
    "Analysis History"
)


if st.session_state.history:

    st.dataframe(
        st.session_state.history,
        use_container_width=True
    )

else:

    st.info(
        "No analysis history yet."
    )


# =========================================================
# DOWNLOAD REPORT
# =========================================================

if st.session_state.last_prediction:

    st.divider()

    st.subheader(
        "Download Analysis Report"
    )

    report = f"""
SMART PLANT DISEASE DETECTOR
========================================

PLANT SELECTED
{plant}

AI PREDICTION
{st.session_state.last_prediction}

CONFIDENCE
{st.session_state.last_confidence * 100:.2f}%

ANALYSIS DATE
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


GENERAL RECOMMENDATIONS
========================================

Inspect the plant and nearby plants.

Keep the growing area clean.

Monitor the plant regularly.

Improve air circulation.

Avoid unnecessary wetting of leaves.

Consult an agricultural expert for confirmation.


IMPORTANT NOTICE
========================================

This AI system provides a supporting prediction.
It does not replace professional agricultural diagnosis.

Confirm important disease diagnoses with a qualified
agricultural professional.


SMART PLANT DISEASE DETECTOR
AI FOR SMART AGRICULTURE
========================================
"""

    st.download_button(
        label="Download Analysis Report",
        data=report,
        file_name="plant_disease_report.txt",
        mime="text/plain",
        use_container_width=True
    )


# =========================================================
# ABOUT
# =========================================================

st.divider()

st.subheader(
    "About the Project"
)

st.write(
    "The Smart Plant Disease Detector is an Artificial "
    "Intelligence project designed to support smart agriculture."
)

st.write(
    "The system analyzes plant leaf images and provides "
    "possible disease predictions, confidence scores and "
    "general recommendations."
)

st.write(
    "The project is designed as a supporting tool for "
    "students, farmers and agricultural users."
)

st.warning(
    "The AI is a supporting tool and should not replace "
    "professional agricultural diagnosis."
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Smart Plant Disease Detector | AI for Smart Agriculture"
)
