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


st.title("🌿 Smart Plant Disease Detector")

st.write(
    "Upload a plant leaf image and the AI will analyze possible diseases."
)

st.warning(
    "AI predictions are not a replacement for agricultural experts."
)


MODEL_URL = (
    "https://huggingface.co/animeshakr/"
    "plant-disease-efficientnetv2s/resolve/main/"
    "model_float16_quant.tflite"
)

MODEL_PATH = "plant_model.tflite"


CLASS_NAMES = [
    "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",
    "Blueberry - Healthy",
    "Cherry - Powdery Mildew",
    "Cherry - Healthy",
    "Corn - Cercospora Leaf Spot",
    "Corn - Common Rust",
    "Corn - Northern Leaf Blight",
    "Corn - Healthy",
    "Grape - Black Rot",
    "Grape - Esca",
    "Grape - Leaf Blight",
    "Grape - Healthy",
    "Orange - Huanglongbing",
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


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading AI model..."):

            r = requests.get(
                MODEL_URL,
                timeout=300
            )

            r.raise_for_status()

            with open(
                MODEL_PATH,
                "wb"
            ) as f:
                f.write(r.content)


    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    return (
        interpreter,
        interpreter.get_input_details(),
        interpreter.get_output_details()
    )


def predict(image):

    interpreter, inputs, outputs = load_model()

    size = inputs[0]["shape"]

    img = image.resize(
        (
            size[2],
            size[1]
        )
    )


    img = np.array(img).astype(
        np.float32
    )


    img = img / 255.0


    img = np.expand_dims(
        img,
        axis=0
    )


    interpreter.set_tensor(
        inputs[0]["index"],
        img
    )

    interpreter.invoke()


    result = interpreter.get_tensor(
        outputs[0]["index"]
    )[0]


    probabilities = tf.nn.softmax(
        result
    ).numpy()


    indexes = probabilities.argsort()[-3:][::-1]


    predictions=[]


    for i in indexes:

        predictions.append(
            (
                CLASS_NAMES[i],
                probabilities[i]
            )
        )


    return predictions



uploaded = st.file_uploader(
    "Upload leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded:


    image = Image.open(
        uploaded
    ).convert("RGB")


    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )


    if st.button(
        "ANALYZE LEAF"
    ):


        try:

            with st.spinner(
                "AI analysing..."
            ):


                results = predict(
                    image
                )


            st.divider()

            st.subheader(
                "AI Prediction"
            )


            best = results[0]


            st.write(
                "Prediction:",
                best[0]
            )


            st.write(
                "Confidence:",
                f"{best[1]*100:.2f}%"
            )


            st.write(
                "Analysis Time:",
                datetime.now()
            )


            st.subheader(
                "Top 3 Results"
            )


            for name,score in results:

                st.write(
                    f"{name} : {score*100:.2f}%"
                )


            if "Healthy" in best[0]:

                st.success(
                    "The plant appears healthy."
                )

            else:

                st.error(
                    "Possible disease detected."
                )


                st.subheader(
                    "Recommendations"
                )


                st.write(
                    """
                    - Remove severely affected leaves.
                    - Keep the farm area clean.
                    - Avoid excessive leaf wetness.
                    - Monitor nearby plants.
                    - Seek expert confirmation before treatment.
                    """
                )


        except Exception as e:

            st.error(
                "Prediction failed"
            )

            st.code(
                str(e)
            )


else:

    st.info(
        "Upload a leaf image to start."
    )


st.divider()


st.subheader(
    "About the Project"
)


st.write(
    """
    Smart Plant Disease Detector is an AI-based agriculture
    project that analyzes plant leaf images and provides
    possible disease predictions.
    """
)
