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
# LANGUAGE SELECTION
# =========================================================

language = st.selectbox(
    "🌐 Choose Language / Londa Olulimi",
    [
        "English",
        "Luganda"
    ]
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

    image_instruction = (
        "📸 For best results, use a clear close-up image "
        "of one leaf with good lighting."
    )

    choose_image = "📷 Choose Your Leaf Image"

    upload_image = "Upload Image"

    use_camera = "Use Camera"

    upload_instruction = "Upload a clear plant leaf image"

    camera_instruction = "Take a clear picture of the leaf"

    analyze = "🔍 ANALYZE LEAF"

    prediction_title = "🌿 AI Prediction"

    confidence = "Confidence"

    analysis_time = "Analysis Time"

    top_predictions = "🔎 Top 5 AI Predictions"

    clear_advice = "💡 Clear Advice"

    download_report = "📄 Download Analysis Report"

    analysis_complete = "✅ Analysis Complete!"

    low_confidence = "⚠️ LOW CONFIDENCE"

    moderate_confidence = "⚠️ MODERATE CONFIDENCE"

    higher_confidence = "✅ HIGHER CONFIDENCE"

    low_confidence_message = (
        "The AI is uncertain about this result. "
        "The prediction may be incorrect."
    )

    moderate_confidence_message = (
        "The AI has some uncertainty about this result."
    )

    higher_confidence_message = (
        "The AI has stronger confidence in this prediction."
    )

    about_title = "📖 About the Project"

    about_text = (
        "The Smart Plant Disease Detector is an Artificial "
        "Intelligence project designed to support smart agriculture. "
        "The system analyzes images of plant leaves and provides "
        "possible disease predictions, confidence scores and "
        "general recommendations."
    )

    important_notice = (
        "⚠️ This AI system provides a supporting prediction. "
        "It does not replace professional agricultural diagnosis."
    )

    no_image = (
        "📷 Upload or capture a clear plant leaf image "
        "to begin analysis."
    )

else:

    title = "🌿 Pulogulaamu Eyekenneenya Obulwadde bw'Ebimera"

    description = (
        "Teekamu oba kwata ekifaananyi ekitegeerekeka obulungi "
        "eky'ekikoola ky'ekimera. AI ejja kwekenneenya ekifaananyi "
        "n'okuwa obulwadde obuyinza okubaawo, obwesige bw'obuvumbuzi "
        "n'amagezi agayinza okukuyamba."
    )

    image_instruction = (
        "📸 Okufuna ebivaamu ebirungi, kozesa ekifaananyi "
        "ekitegeerekeka obulungi eky'ekikoola kimu nga waliwo "
        "ekitangaala ekimala."
    )

    choose_image = "📷 Londa Ekifaananyi ky'Ekikoola"

    upload_image = "Teekamu Ekifaananyi"

    use_camera = "Kozesa Camera"

    upload_instruction = (
        "Teekamu ekifaananyi ekitegeerekeka obulungi "
        "eky'ekikoola ky'ekimera"
    )

    camera_instruction = (
        "Kwata ekifaananyi ekitegeerekeka obulungi "
        "eky'ekikoola"
    )

    analyze = "🔍 KEENNEENYA EKIKOOLO"

    prediction_title = "🌿 Obuvumbuzi bwa AI"

    confidence = "Obwesige"

    analysis_time = "Obudde Obw'okwekeneenya"

    top_predictions = (
        "🔎 Obuvumbuzi 5 Obusinga Okuba Waggulu"
    )

    clear_advice = "💡 Amagezi Amakulu"

    download_report = "📄 Wanula Lipoota y'Okwekeneenya"

    analysis_complete = "✅ Okwekeneenya Kuwedde!"

    low_confidence = "⚠️ OBWESIGE BUKYALI BUTONO"

    moderate_confidence = "⚠️ OBWESIGE BWA WAKATI"

    higher_confidence = "✅ OBWESIGE BUKULU"

    low_confidence_message = (
        "AI tekwesiga nnyo bivudde mu kwekenneenya kuno. "
        "Obuvumbuzi buyinza okuba nga si butuufu."
    )

    moderate_confidence_message = (
        "AI erina obutali bukakafu obumu ku bivudde mu kwekenneenya kuno."
    )

    higher_confidence_message = (
        "AI erina obwesige obusingako mu buvumbuzi buno."
    )

    about_title = "📖 Ebikwata ku Pulogulaamu"

    about_text = (
        "Pulogulaamu Eyekenneenya Obulwadde bw'Ebimera ye "
        "pulojekiti ya Artificial Intelligence eyakolebwa okuyamba "
        "mu by'obulimi. Enkola eno ekebera ebifaananyi by'ebikoola "
        "by'ebimera n'okuwa obulwadde obuyinza okubaawo, obwesige "
        "bw'obuvumbuzi n'amagezi agayinza okuyamba."
    )

    important_notice = (
        "⚠️ Enkola eno ewa buvumbuzi bwa buyambi bwokka. "
        "Teddira kifo kya kwekenneenya kwa mukugu mu by'obulimi."
    )

    no_image = (
        "📷 Teekamu oba kwata ekifaananyi ekitegeerekeka "
        "obulungi eky'ekikoola okutandika okwekeneenya."
    )


# =========================================================
# TITLE
# =========================================================

st.title(title)

st.write(description)

st.info(image_instruction)

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
    # This assumes the model expects pixel values
    # from 0 to 255.
    #
    # If your specific model expects 0 to 1,
    # change this to:
    #
    # image_array = image_array / 255.0

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
# ADVICE FUNCTION
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
            low_confidence
        )

        st.write(
            low_confidence_message
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
            "📱 Avoid blurry photographs."
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
            "⚠️ Do not apply chemical treatment "
            "based only on a low-confidence prediction."
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
    # GRAPE CONDITIONS
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
    # CORN CONDITIONS
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
    # POTATO CONDITIONS
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
    # GENERAL CONDITION
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
    choose_image
)


input_method = st.radio(
    "Select an option / Londa ekikozesebwa:",
    [
        upload_image,
        use_camera
    ],
    horizontal=True
)


uploaded_file = None


# =========================================================
# UPLOAD IMAGE
# =========================================================

if input_method == upload_image:

    uploaded_file = st.file_uploader(
        upload_instruction,
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
        camera_instruction
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
        caption="Uploaded Plant Leaf",
        use_container_width=True
    )


    st.divider()


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


            if len(results) == 0:

                st.error(
                    "No prediction was produced."
                )

                st.stop()


            # =================================================
            # BEST PREDICTION
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
            # MAIN RESULT
            # =================================================

            st.success(
                analysis_complete
            )


            st.subheader(
                prediction_title
            )


            st.write(
                f"**Prediction:** {prediction}"
            )


            st.write(
                f"**{confidence}:** "
                f"{confidence * 100:.2f}%"
            )


            st.write(
                f"**{analysis_time}:** "
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

            st.divider()


            if confidence < 0.40:

                st.error(
                    low_confidence
                )

                st.write(
                    low_confidence_message
                )


            elif confidence < 0.70:

                st.warning(
                    moderate_confidence
                )

                st.write(
                    moderate_confidence_message
                )


            else:

                st.success(
                    higher_confidence
                )

                st.write(
                    higher_confidence_message
                )


            # =================================================
            # TOP 5 PREDICTIONS
            # =================================================

            st.divider()


            st.subheader(
                top_predictions
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
                clear_advice
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
                download_report
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

IMPORTANT NOTICE
----------------------------------------

{important_notice}

Confirm the diagnosis with a qualified
agricultural professional before making
important crop management decisions.


SMART PLANT DISEASE DETECTOR
AI FOR SMART AGRICULTURE
========================================
"""


            st.download_button(
                label=download_report,
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
        no_image
    )


# =========================================================
# ABOUT THE PROJECT
# =========================================================

st.divider()


st.subheader(
    about_title
)


st.write(
    about_text
)


st.write(
    important_notice
)


# =========================================================
# FOOTER
# =========================================================

st.divider()


if language == "English":

    st.caption(
        "🌿 Smart Plant Disease Detector | "
        "AI for Smart Agriculture"
    )

else:

    st.caption(
        "🌿 Pulogulaamu Eyekenneenya Obulwadde "
        "bw'Ebimera | AI mu Bulimi"
    )
