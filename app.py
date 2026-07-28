
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os
from datetime import datetime

st.set_page_config(
    page_title="Smart Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("Smart Plant Disease Detector")

st.write(
    "Upload a clear image of a plant leaf. "
    "The AI will analyze the image and provide a possible prediction."
)

st.warning(
    "AI results are predictions only. "
    "For important agricultural decisions, confirm the result with an expert."
)

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


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading AI model..."):

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


def predict_image(image):

    (
        interpreter,
        input_details,
        output_details
    ) = load_model()

    input_shape = input_details[0]["shape"]

    height = int(input_shape[1])
    width = int(input_shape[2])

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
        input_details[0]["index"],
        image_array
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    predictions = output[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_index]
    )

    if predicted_index < len(CLASS_NAMES):

        prediction = CLASS_NAMES[
            predicted_index
        ]

    else:

        prediction = "Unknown"

    return prediction, confidence


st.divider()

selected_plant = st.selectbox(
    "Select the plant you are analyzing",
    SUPPORTED_PLANTS
)

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Plant Leaf",
        use_container_width=True
    )

    if st.button(
        "ANALYZE LEAF",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "AI is analyzing the leaf..."
            ):

                prediction, confidence = predict_image(
                    image
                )

            predicted_plant = prediction.split(
                " - "
            )[0]

            analysis_time = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

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

            if predicted_plant != selected_plant:

                st.error(
                    "PLANT MISMATCH"
                )

                st.write(
                    "The AI predicted "
                    + predicted_plant
                    + " but you selected "
                    + selected_plant
                    + "."
                )

                st.write(
                    "This result may be incorrect. "
                    "Try another clear image."
                )

            elif confidence < 0.40:

                st.warning(
                    "LOW CONFIDENCE"
                )

                st.write(
                    "The AI is not confident about this prediction."
                )

                st.write(
                    "Try taking another clear photograph "
                    "with good lighting."
                )

            elif "Healthy" in prediction:

                st.success(
                    "The AI predicts that the plant may be healthy."
                )

                st.write(
                    "Continue monitoring the plant regularly."
                )

            else:

                st.error(
                    "POSSIBLE DISEASE DETECTED"
                )

                st.write(
                    "The AI has detected a possible "
                    + prediction
                    + " condition."
                )

                st.subheader(
                    "Recommended Actions"
                )

                st.write(
                    "Inspect other leaves on the same plant."
                )

                st.write(
                    "Check nearby plants for similar symptoms."
                )

                st.write(
                    "Remove severely affected plant material "
                    "where appropriate."
                )

                st.write(
                    "Keep the growing area clean."
                )

                st.write(
                    "Improve air circulation around the plant."
                )

                st.write(
                    "Avoid unnecessary wetting of leaves."
                )

                st.write(
                    "Monitor new leaves for additional symptoms."
                )

                st.write(
                    "Consult an agricultural expert for confirmation."
                )

                st.warning(
                    "Confirm the diagnosis before applying "
                    "any chemical treatment."
                )

        except Exception as error:

            st.error(
                "An error occurred while analyzing the image."
            )

            st.code(
                str(error)
            )

else:

    st.info(
        "Upload a clear plant leaf image to begin."
    )


st.divider()

st.subheader(
    "About the Project"
)

st.write(
    "The Smart Plant Disease Detector is an Artificial "
    "Intelligence project designed to support smart agriculture."
)

st.write(
    "The system analyzes images of plant leaves and provides "
    "possible disease predictions, confidence scores and "
    "general recommendations."
)

st.write(
    "The system is designed as a supporting tool and should "
    "not replace professional agricultural diagnosis."
)

st.caption(
    "Smart Plant Disease Detector | AI for Smart Agriculture"
)


