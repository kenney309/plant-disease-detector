```python
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
    page_title="Smart Plant Doctor",
    page_icon=None,
    layout="centered"
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
# IMPORTANT:
# These must match the exact order used when the model
# was trained.
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
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
    <style>

    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #cccccc;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">Smart Plant Disease Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI for Smart Agriculture</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload or capture a clear image of a plant leaf. "
    "The AI will analyze the image and provide a possible "
    "plant health prediction."
)

st.warning(
    "This system provides a possible AI prediction. "
    "It should not replace professional agricultural diagnosis."
)

st.divider()


# =========================================================
# PLANT SELECTION
# =========================================================

st.subheader("Select Plant")

PLANTS = [
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

selected_plant = st.selectbox(
    "Which plant are you analyzing?",
    PLANTS
)


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

    width, height = image.size

    if width < 200 or height < 200:

        return (
            False,
            "The image is too small. Please use a clearer image."
        )

    gray = image.convert("L")

    brightness = ImageStat.Stat(
        gray
    ).mean[0]

    if brightness < 35:

        return (
            False,
            "The image is too dark. Take the photo in better lighting."
        )

    if brightness > 245:

        return (
            False,
            "The image is too bright. Reduce excessive lighting."
        )

    return (
        True,
        "Image quality looks suitable for analysis."
    )


# =========================================================
# AI PREDICTION
# =========================================================

def predict_image(image):

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
        (width, height)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # IMPORTANT:
    # This assumes the model expects values from 0 to 1.
    # If the original model was trained with another
    # preprocessing method, this must be changed.

    image_array = (
        image_array / 255.0
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

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    predictions = output[0]

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    # Convert logits to probabilities if necessary

    if (
        np.min(predictions) < 0
        or np.max(predictions) > 1
        or not np.isclose(
            np.sum(predictions),
            1.0,
            atol=0.05
        )
    ):

        exp_values = np.exp(
            predictions - np.max(predictions)
        )

        predictions = (
            exp_values
            / np.sum(exp_values)
        )

    top_indices = np.argsort(
        predictions
    )[::-1][:5]

    results = []

    for index in top_indices:

        if index < len(CLASS_NAMES):

            results.append(
                (
                    CLASS_NAMES[index],
                    float(
                        predictions[index]
                    )
                )
            )

    return results


# =========================================================
# RECOMMENDATIONS
# =========================================================

def show_recommendations():

    st.write(
        "Inspect the affected plant and nearby plants."
    )

    st.write(
        "Remove severely affected plant material "
        "where appropriate."
    )

    st.write(
        "Keep the growing area clean."
    )

    st.write(
        "Improve air circulation around the plants."
    )

    st.write(
        "Avoid unnecessary wetting of leaves."
    )

    st.write(
        "Monitor new symptoms regularly."
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
        "Inspect plants regularly."
    )

    st.write(
        "Keep the growing area clean."
    )

    st.write(
        "Maintain good spacing between plants."
    )

    st.write(
        "Avoid unnecessary moisture on leaves."
    )


# =========================================================
# IMAGE INPUT
# =========================================================

st.divider()

st.subheader(
    "Leaf Image"
)

input_method = st.radio(
    "Choose image source",
    [
        "Upload Image",
        "Use Camera"
    ]
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
        "Take a photo of the leaf"
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


    if st.button(
        "ANALYZE LEAF",
        use_container_width=True
    ):

        if not quality_ok:

            st.error(
                "Please upload a clearer image before analysis."
            )

        else:

            try:

                with st.spinner(
                    "AI is analyzing the leaf..."
                ):

                    results = predict_image(
                        image
                    )


                if not results:

                    st.error(
                        "The AI did not return a prediction."
                    )

                    st.stop()


                prediction = (
                    results[0][0]
                )

                confidence = (
                    results[0][1]
                )

                analysis_time = (
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )


                # =========================================
                # RESULT
                # =========================================

                st.divider()

                st.subheader(
                    "AI Prediction"
                )

                st.write(
                    "Plant Selected: "
                    + selected_plant
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


                # =========================================
                # PLANT MISMATCH CHECK
                # =========================================

                predicted_plant = (
                    prediction.split(
                        " - "
                    )[0]
                )

                if (
                    predicted_plant
                    != selected_plant
                ):

                    st.error(
                        "PLANT MISMATCH"
                    )

                    st.write(
                        "You selected "
                        + selected_plant
                        + ", but the AI predicted "
                        + predicted_plant
                        + "."
                    )

                    st.write(
                        "This prediction may be unreliable. "
                        "Please check the image and try again."
                    )


                # =========================================
                # CONFIDENCE WARNING
                # =========================================

                if confidence < 0.40:

                    st.error(
                        "LOW CONFIDENCE"
                    )

                    st.write(
                        "The AI is not confident about this result."
                    )

                    st.write(
                        "The prediction may be incorrect."
                    )

                    st.write(
                        "Try another clear photograph."
                    )

                elif confidence < 0.70:

                    st.warning(
                        "MODERATE CONFIDENCE"
                    )

                    st.write(
                        "The AI has some uncertainty about this result."
                    )

                else:

                    st.success(
                        "HIGHER CONFIDENCE"
                    )


                # =========================================
                # HEALTH STATUS
                # =========================================

                st.divider()

                st.subheader(
                    "Plant Health Status"
                )

                if "Healthy" in prediction:

                    status = "Likely Healthy"

                elif confidence >= 0.70:

                    status = "Possible Disease Detected"

                elif confidence >= 0.40:

                    status = "Needs Further Investigation"

                else:

                    status = "Unable to Confirm"


                st.write(
                    "Status: "
                    + status
                )


                # =========================================
                # TOP 5 PREDICTIONS
                # =========================================

                st.divider()

                st.subheader(
                    "Top 5 AI Predictions"
                )

                for number, (
                    name,
                    score
                ) in enumerate(
                    results,
                    start=1
                ):

                    st.write(
                        str(number)
                        + ". "
                        + name
                        + " - "
                        + f"{score * 100:.2f}%"
                    )


                # =========================================
                # RECOMMENDATIONS
                # =========================================

                st.divider()

                st.subheader(
                    "Recommended Actions"
                )

                show_recommendations()


                # =========================================
                # PREVENTION
                # =========================================

                st.divider()

                st.subheader(
                    "Prevention Tips"
                )

                show_prevention()


                # =========================================
                # SAVE RESULT
                # =========================================

                st.session_state.last_result = {
                    "plant": selected_plant,
                    "prediction": prediction,
                    "confidence": confidence,
                    "time": analysis_time
                }

                st.session_state.history.append(
                    {
                        "Time": analysis_time,
                        "Plant": selected_plant,
                        "Prediction": prediction,
                        "Confidence":
                            f"{confidence * 100:.2f}%",
                        "Status": status
                    }
                )


            except Exception as error:

                st.error(
                    "An error occurred during analysis."
                )

                st.code(
                    str(error)
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

if st.session_state.last_result:

    st.divider()

    st.subheader(
        "Download Analysis Report"
    )

    result = (
        st.session_state.last_result
    )

    report = f"""
SMART PLANT DISEASE DETECTOR
====================================

PLANT SELECTED:
{result["plant"]}

AI PREDICTION:
{result["prediction"]}

CONFIDENCE:
{result["confidence"] * 100:.2f}%

ANALYSIS TIME:
{result["time"]}

RECOMMENDATIONS:
- Inspect the affected plant.
- Inspect nearby plants.
- Keep the growing area clean.
- Improve air circulation.
- Avoid unnecessary wetting of leaves.
- Monitor the plant regularly.
- Consult an agricultural expert for confirmation.

IMPORTANT:
This AI prediction is for support and information.
It does not replace professional agricultural diagnosis.
"""

    st.download_button(
        "Download Analysis Report",
        data=report,
        file_name="plant_disease_analysis.txt",
        mime="text/plain",
        use_container_width=True
    )


# =========================================================
# OFFLINE PLANT GUIDE
# =========================================================

st.divider()

st.subheader(
    "Offline Plant Health Guide"
)

st.write(
    "Use this section to access basic plant health "
    "information even when you do not need an AI prediction."
)

offline_plant = st.selectbox(
    "Choose a plant",
    PLANTS,
    key="offline_plant"
)


OFFLINE_INFO = {

    "Apple":
        "Common conditions include apple scab, black rot "
        "and cedar apple rust.",

    "Blueberry":
        "Monitor leaves and fruit regularly for unusual "
        "changes or symptoms.",

    "Cherry":
        "Powdery mildew is one condition included in "
        "the current AI model.",

    "Corn":
        "The model includes common rust, northern leaf "
        "blight and cercospora leaf spot.",

    "Grape":
        "The model includes black rot, Esca and leaf blight.",

    "Orange":
        "The model includes Huanglongbing, also known as "
        "citrus greening.",

    "Peach":
        "The model includes bacterial spot.",

    "Pepper":
        "The model includes bacterial spot.",

    "Potato":
        "The model includes early blight and late blight.",

    "Raspberry":
        "Monitor plants regularly for unusual symptoms.",

    "Soybean":
        "Monitor plants regularly and maintain good "
        "field hygiene.",

    "Squash":
        "The model includes powdery mildew.",

    "Strawberry":
        "The model includes leaf scorch.",

    "Tomato":
        "The model includes bacterial spot, early blight, "
        "late blight, leaf mold, septoria leaf spot, "
        "spider mites, target spot and viral diseases."
}


st.write(
    OFFLINE_INFO[offline_plant]
)

st.warning(
    "This information is general guidance and does not "
    "replace professional diagnosis."
)


# =========================================================
# PLANT DOCTOR
# =========================================================

st.divider()

st.subheader(
    "Plant Doctor"
)

st.write(
    "Ask a general question about plant health."
)

question = st.text_input(
    "Your question"
)

if st.button(
    "GET ADVICE",
    use_container_width=True
):

    if question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        q = question.lower()

        if (
            "prevent" in q
            or "avoid" in q
        ):

            st.write(
                "Use healthy planting material, inspect "
                "plants regularly, maintain good spacing, "
                "keep the growing area clean and avoid "
                "unnecessary moisture on leaves."
            )

        elif (
            "treat" in q
            or "treatment" in q
        ):

            st.write(
                "Confirm the diagnosis before applying "
                "any treatment. Inspect nearby plants, "
                "remove severely affected material where "
                "appropriate and consult an agricultural expert."
            )

        elif (
            "spread" in q
        ):

            st.write(
                "Some plant diseases can spread through "
                "water, insects, contaminated tools, infected "
                "plant material and environmental conditions."
            )

        else:

            st.write(
                "Carefully inspect the plant, monitor "
                "nearby plants and seek professional "
                "agricultural advice when necessary."
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
    "The project is intended as a supporting tool for "
    "students, farmers and agricultural users."
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Smart Plant Disease Detector | AI for Smart Agriculture"
)
```
