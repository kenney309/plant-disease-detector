```python
import streamlit as st
from PIL import Image, ImageStat
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
# SESSION STATE - ANALYSIS HISTORY
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
# LANGUAGE SELECTION
# =========================================================

language = st.selectbox(
    "🌐 Choose Language / Londa Olulimi",
    ["English", "Luganda"]
)


# =========================================================
# TRANSLATIONS
# =========================================================

if language == "English":

    title = "🌿 Smart Plant Disease Detector"

    description = (
        "Upload or capture a clear image of a plant leaf. "
        "The AI will analyze the image and provide a possible "
        "plant disease prediction, confidence score and advice."
    )

    image_tip = (
        "📸 For best results, use a clear close-up image "
        "of one leaf with good lighting."
    )

    choose_plant = "🌱 Select the Plant"

    choose_image = "📷 Choose Your Leaf Image"

    upload = "Upload Image"

    camera = "Use Camera"

    analyze = "🔍 ANALYZE LEAF"

    prediction_title = "🌿 AI Prediction"

    top_predictions = "🔎 Top 5 AI Predictions"

    disease_info = "📚 Disease Information"

    advice = "💡 Management Advice"

    prevention = "🛡️ Prevention Tips"

    history_title = "📜 Analysis History"

    about = "📖 About the Project"

    download = "📄 Download Analysis Report"

    complete = "✅ Analysis Complete!"

else:

    title = "🌿 Pulogulaamu Eyekenneenya Obulwadde bw'Ebimera"

    description = (
        "Teekamu oba kwata ekifaananyi ekitegeerekeka obulungi "
        "eky'ekikoola ky'ekimera. AI ejja kwekenneenya ekifaananyi "
        "n'okuwa obuvumbuzi n'amagezi."
    )

    image_tip = (
        "📸 Okufuna ebivaamu ebirungi, kozesa ekifaananyi "
        "ekitegeerekeka obulungi eky'ekikoola kimu."
    )

    choose_plant = "🌱 Londa Ekika ky'Ekimera"

    choose_image = "📷 Londa Ekifaananyi ky'Ekikoola"

    upload = "Teekamu Ekifaananyi"

    camera = "Kozesa Camera"

    analyze = "🔍 KEENNEENYA EKIKOOLO"

    prediction_title = "🌿 Obuvumbuzi bwa AI"

    top_predictions = "🔎 Obuvumbuzi 5 Obusinga Okuba Waggulu"

    disease_info = "📚 Ebikwata ku Bulwadde"

    advice = "💡 Amagezi g'Okulabirira Ekimera"

    prevention = "🛡️ Engeri y'Okwewala Obulwadde"

    history_title = "📜 Eby'okwekeneenya ebyayita"

    about = "📖 Ebikwata ku Pulojekiti"

    download = "📄 Wanula Lipoota"

    complete = "✅ Okwekeneenya Kuwedde!"


# =========================================================
# TITLE
# =========================================================

st.title(title)

st.write(description)

st.info(image_tip)

st.divider()


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


plant = st.selectbox(
    choose_plant,
    SUPPORTED_PLANTS
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
# MODEL LOADING
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
# IMAGE QUALITY CHECK
# =========================================================

def check_image_quality(image):

    gray_image = image.convert("L")

    statistics = ImageStat.Stat(
        gray_image
    )

    brightness = statistics.mean[0]

    if brightness < 35:

        return (
            False,
            "⚠️ The image appears too dark. "
            "Please take another photo in better lighting."
        )

    if brightness > 245:

        return (
            False,
            "⚠️ The image appears too bright. "
            "Please avoid excessive light."
        )

    width, height = image.size

    if width < 200 or height < 200:

        return (
            False,
            "⚠️ The image resolution is low. "
            "Please use a clearer, higher-resolution image."
        )

    return (
        True,
        "✅ Image quality appears suitable for analysis."
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

    input_shape = (
        input_details[0]["shape"]
    )

    height = int(
        input_shape[1]
    )

    width = int(
        input_shape[2]
    )

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

    # IMPORTANT:
    # This preprocessing must match
    # the preprocessing used to train
    # the original model.

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

    # Convert logits/probabilities safely
    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    # If output does not look like probabilities,
    # apply softmax.
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
            predictions
            - np.max(predictions)
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
# DISEASE INFORMATION
# =========================================================

DISEASE_INFO = {

    "Apple - Apple Scab": (
        "Apple scab is a fungal disease that can affect "
        "apple leaves and fruit. It commonly causes dark "
        "or olive-colored spots."
    ),

    "Apple - Black Rot": (
        "Black rot is a fungal disease that can affect "
        "apple leaves, fruit and branches."
    ),

    "Apple - Cedar Apple Rust": (
        "Cedar apple rust is a fungal disease that can "
        "cause yellow or orange-colored symptoms on leaves."
    ),

    "Corn - Common Rust": (
        "Common rust is a fungal disease of corn that "
        "can produce reddish-brown rust-like spots."
    ),

    "Corn - Northern Leaf Blight": (
        "Northern leaf blight is a fungal disease that "
        "can cause long gray or brown lesions on corn leaves."
    ),

    "Tomato - Early Blight": (
        "Early blight is a fungal disease that can cause "
        "dark spots and lesions on tomato leaves."
    ),

    "Tomato - Late Blight": (
        "Late blight is a serious plant disease that can "
        "affect tomato and potato crops."
    ),

    "Tomato - Leaf Mold": (
        "Tomato leaf mold is a fungal disease that commonly "
        "affects leaves under humid conditions."
    ),

    "Potato - Early Blight": (
        "Potato early blight can cause dark lesions on "
        "potato leaves and may reduce plant productivity."
    ),

    "Potato - Late Blight": (
        "Potato late blight is a serious disease that "
        "can spread rapidly under favorable conditions."
    ),

    "Grape - Black Rot": (
        "Grape black rot is a fungal disease that can "
        "affect grape leaves and fruit."
    ),

    "Grape - Esca (Black Measles)": (
        "Esca is a complex grapevine disease associated "
        "with wood-inhabiting fungi."
    ),

    "Grape - Leaf Blight": (
        "Grape leaf blight can cause damage to grape leaves "
        "and may affect plant growth."
    )
}


# =========================================================
# GENERAL MANAGEMENT ADVICE
# =========================================================

def show_advice(prediction):

    st.write(
        "🔍 Inspect the plant and nearby plants "
        "for similar symptoms."
    )

    st.write(
        "🍂 Remove severely affected plant material "
        "where appropriate."
    )

    st.write(
        "🧹 Keep the growing area clean."
    )

    st.write(
        "🌬️ Improve air circulation around plants."
    )

    st.write(
        "💧 Avoid unnecessary wetting of leaves."
    )

    st.write(
        "👀 Monitor the plant regularly for changes."
    )

    st.write(
        "👨‍🌾 Consult an agricultural expert "
        "for confirmation."
    )


# =========================================================
# PREVENTION ADVICE
# =========================================================

def show_prevention():

    st.write(
        "🌱 Use healthy planting materials."
    )

    st.write(
        "🧹 Keep the field or garden clean."
    )

    st.write(
        "🔍 Inspect plants regularly."
    )

    st.write(
        "🌬️ Maintain good spacing and air circulation."
    )

    st.write(
        "💧 Avoid unnecessary moisture on leaves."
    )

    st.write(
        "👨‍🌾 Follow advice from agricultural professionals."
    )


# =========================================================
# IMAGE INPUT
# =========================================================

st.subheader(
    choose_image
)

input_method = st.radio(
    "Select an option:",
    [
        upload,
        camera
    ],
    horizontal=True
)

uploaded_file = None


if input_method == upload:

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
# ANALYSIS
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


    # =====================================================
    # IMAGE QUALITY
    # =====================================================

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


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    if st.button(
        analyze,
        use_container_width=True
    ):

        try:

            with st.spinner(
                "🤖 AI is analyzing the leaf..."
            ):

                results = predict_leaf(
                    image
                )


            if not results:

                st.error(
                    "No prediction was produced."
                )

                st.stop()


            # =================================================
            # BEST RESULT
            # =================================================

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
            # CHECK PLANT MATCH
            # =================================================

            predicted_plant = (
                prediction.split(" - ")[0]
            )


            if predicted_plant != plant:

                st.warning(
                    f"⚠️ You selected **{plant}**, "
                    f"but the AI prediction belongs to "
                    f"**{predicted_plant}**."
                )

                st.write(
                    "The result may be unreliable. "
                    "Make sure the selected plant matches "
                    "the uploaded leaf."
                )


            # =================================================
            # MAIN RESULT
            # =================================================

            st.success(
                complete
            )

            st.subheader(
                prediction_title
            )

            st.write(
                f"🌱 **Plant Selected:** {plant}"
            )

            st.write(
                f"🔬 **Prediction:** {prediction}"
            )

            st.write(
                f"🎯 **Confidence:** "
                f"{confidence * 100:.2f}%"
            )

            st.write(
                f"🕒 **Analysis Time:** "
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
            # CONFIDENCE WARNING
            # =================================================

            if confidence < 0.40:

                st.error(
                    "⚠️ LOW CONFIDENCE"
                )

                st.write(
                    "The AI is not confident about this prediction. "
                    "The result may be incorrect."
                )

                st.write(
                    "📸 Try another clear image."
                )

                st.write(
                    "☀️ Use good natural lighting."
                )

                st.write(
                    "🔍 Make sure the leaf is clearly visible."
                )

                st.write(
                    "👨‍🌾 Consult an agricultural expert "
                    "for confirmation."
                )

            elif confidence < 0.70:

                st.warning(
                    "⚠️ MODERATE CONFIDENCE"
                )

                st.write(
                    "The AI has some uncertainty about this result."
                )

            else:

                st.success(
                    "✅ HIGHER CONFIDENCE"
                )


            # =================================================
            # TOP 5 PREDICTIONS
            # =================================================

            st.divider()

            st.subheader(
                top_predictions
            )


            chart_data = {}

            for number, result in enumerate(
                results,
                start=1
            ):

                st.write(
                    f"**{number}. "
                    f"{result['name']}** — "
                    f"{result['confidence'] * 100:.2f}%"
                )

                chart_data[
                    result["name"]
                ] = result["confidence"]


            st.bar_chart(
                chart_data
            )


            # =================================================
            # DISEASE INFORMATION
            # =================================================

            st.divider()

            st.subheader(
                disease_info
            )


            if "Healthy" in prediction:

                st.success(
                    "🌿 The AI prediction indicates "
                    "that the plant may be healthy."
                )

                st.write(
                    "Continue monitoring the plant regularly "
                    "for any new symptoms."
                )

            elif prediction in DISEASE_INFO:

                st.write(
                    DISEASE_INFO[prediction]
                )

            else:

                st.write(
                    "The AI has identified a possible "
                    "plant health condition. More information "
                    "may be required to confirm the diagnosis."
                )


            # =================================================
            # MANAGEMENT ADVICE
            # =================================================

            st.divider()

            st.subheader(
                advice
            )

            show_advice(
                prediction
            )


            # =================================================
            # PREVENTION
            # =================================================

            st.divider()

            st.subheader(
                prevention
            )

            show_prevention()


            # =================================================
            # ANALYSIS HISTORY
            # =================================================

            st.session_state.history.append(
                {
                    "Time":
                        analysis_time,

                    "Plant":
                        plant,

                    "Prediction":
                        prediction,

                    "Confidence":
                        f"{confidence * 100:.2f}%"
                }
            )


            # =================================================
            # DOWNLOAD REPORT
            # =================================================

            st.divider()

            st.subheader(
                download
            )


            report = f"""
SMART PLANT DISEASE DETECTOR
========================================

PLANT SELECTED
----------------------------------------
{plant}

ANALYSIS TIME
----------------------------------------
{analysis_time}

AI PREDICTION
----------------------------------------
{prediction}

CONFIDENCE
----------------------------------------
{confidence * 100:.2f}%


TOP 5 AI PREDICTIONS
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

GENERAL ADVICE
----------------------------------------

- Inspect the plant and nearby plants.
- Keep the growing area clean.
- Monitor the plant regularly.
- Improve air circulation.
- Avoid unnecessary wetting of leaves.
- Consult an agricultural expert for confirmation.


IMPORTANT NOTICE
----------------------------------------

This AI system provides a supporting prediction.
It does not replace professional agricultural
diagnosis.

Confirm important disease diagnoses with a
qualified agricultural professional before
taking major management actions.


SMART PLANT DISEASE DETECTOR
AI FOR SMART AGRICULTURE
========================================
"""


            st.download_button(
                label=download,
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
# ANALYSIS HISTORY
# =========================================================

st.divider()

st.subheader(
    history_title
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
# ABOUT PROJECT
# =========================================================

st.divider()

st.subheader(
    about
)

st.write(
    "The Smart Plant Disease Detector is an "
    "Artificial Intelligence project designed "
    "to support smart agriculture."
)

st.write(
    "The system analyzes plant leaf images and "
    "provides possible disease predictions, "
    "confidence scores and general management advice."
)

st.warning(
    "⚠️ The AI is a supporting tool and should not "
    "replace professional agricultural diagnosis."
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🌿 Smart Plant Disease Detector | "
    "AI for Smart Agriculture"
)
```
