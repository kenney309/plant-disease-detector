import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os
import time
from io import BytesIO


st.set_page_config(
    page_title="Smart Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)


MODEL_URL = "https://huggingface.co/animeshakr/plant-disease-efficientnetv2s/resolve/main/model_float16_quant.tflite"

MODEL_FILE = "plant_model.tflite"


CLASS_NAMES = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Apple Healthy",
    "Corn Leaf Blight",
    "Corn Common Rust",
    "Corn Healthy",
    "Grape Black Rot",
    "Grape Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Healthy"
]


RECOMMENDATIONS = {
    "Apple Scab": "Remove infected leaves, improve air circulation and apply suitable fungicide.",
    "Apple Black Rot": "Remove damaged parts and use proper disease control methods.",
    "Corn Leaf Blight": "Use resistant varieties and avoid excessive moisture.",
    "Corn Common Rust": "Apply recommended fungicides and monitor crop health.",
    "Grape Black Rot": "Remove infected fruits and maintain field hygiene.",
    "Potato Early Blight": "Remove infected leaves and use crop rotation.",
    "Potato Late Blight": "Apply fungicide and avoid water staying on leaves.",
    "Tomato Early Blight": "Remove infected leaves and improve plant spacing.",
    "Tomato Late Blight": "Destroy infected plants and apply treatment.",
    "Healthy": "Your plant appears healthy. Continue proper care."
}


def download_model():

    if not os.path.exists(MODEL_FILE):

        response = requests.get(MODEL_URL)

        with open(MODEL_FILE, "wb") as f:
            f.write(response.content)

    return MODEL_FILE



@st.cache_resource
def load_model():

    model_path = download_model()

    interpreter = tf.lite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()

    return interpreter



def predict(image):

    interpreter = load_model()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    img = image.resize((224,224))

    img = np.array(img)

    img = img.astype(np.float32)/255.0

    img = np.expand_dims(img, axis=0)


    interpreter.set_tensor(
        input_details[0]["index"],
        img
    )

    interpreter.invoke()


    output = interpreter.get_tensor(
        output_details[0]["index"]
    )


    prediction = np.argmax(output)

    confidence = float(np.max(output))*100


    if prediction < len(CLASS_NAMES):
        result = CLASS_NAMES[prediction]
    else:
        result = "Unknown Plant"


    return result, confidence



st.title("🌿 Smart Plant Disease Detector")

st.write(
    "Upload a plant leaf image and the AI model will analyse possible diseases."
)


uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg","jpeg","png"]
)



if uploaded_file:


    image = Image.open(uploaded_file)


    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )


    if st.button("Analyze Plant"):


        start = time.time()


        with st.spinner("AI is analysing the leaf..."):

            disease, confidence = predict(image)



        duration = round(time.time()-start,2)


        st.success("Analysis Complete")


        st.subheader("🌿 AI Prediction")


        st.write("Plant Diagnosis:", disease)

        st.write(
            "Confidence:",
            f"{confidence:.2f}%"
        )


        st.write(
            "Analysis Time:",
            f"{duration} seconds"
        )


        if confidence < 60:

            st.warning(
                "Low confidence. Try uploading a clearer leaf image."
            )


        st.subheader("Recommended Action")


        advice = RECOMMENDATIONS.get(
            disease,
            "Monitor the plant and consult an agricultural expert."
        )


        st.info(advice)



st.divider()


st.subheader("About The Project")

st.write(
"""
Smart Plant Disease Detector is an AI-powered application
that helps farmers and students identify possible plant diseases
using image recognition technology.

The system analyses leaf images and provides a prediction together
with basic recommended actions.
"""
)
