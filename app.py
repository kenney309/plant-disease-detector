import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿"
)

st.title("🌿 Plant Disease Detector")

st.write(
    "Select the plant type, upload a leaf image, "
    "and let the AI analyze it."
)

st.divider()

# ==============================
# MODEL
# ==============================

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

# ==============================
# DOWNLOAD MODEL
# ==============================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        response = requests.get(
            MODEL_URL,
            timeout=300
        )

        response.raise_for_status()

        with open(
            MODEL_PATH,
            "wb"
        ) as f:

            f.write(response.content)

    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    return (
        interpreter,
        interpreter.get_input_details(),
        interpreter.get_output_details()
    )


# ==============================
# PREDICTION
# ==============================

def predict(image):

    interpreter, inputs, outputs = load_model()

    shape = inputs[0]["shape"]

    image = image.resize(
        (shape[2], shape[1])
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
        inputs[0]["index"],
        image_array
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        outputs[0]["index"]
    )[0]

    index = int(
        np.argmax(output)
    )

    confidence = float(
        output[index]
    )

    return (
        CLASS_NAMES[index],
        confidence
    )


# ==============================
# PLANT SELECTION
# ==============================

plant = st.selectbox(
    "🌱 Select the plant you are testing:",
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

if plant == "Guava":

    st.info(
        "🍈 Guava is not included in the current AI model. "
        "The AI result may therefore be inaccurate."
    )

elif plant == "Mango":

    st.info(
        "🥭 Mango is not included in the current AI model. "
        "The AI result may therefore be inaccurate."
    )


# ==============================
# UPLOAD IMAGE
# ==============================

uploaded_file = st.file_uploader(
    "📷 Upload the leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption=f"{plant} Leaf",
        use_container_width=True
    )

    if st.button(
        "🔍 ANALYZE LEAF"
    ):

        with st.spinner(
            "🤖 AI is analyzing..."
        ):

            prediction, confidence = predict(
                image
            )

        st.subheader(
            "🌿 AI Result"
        )

        st.write(
            f"**Selected Plant:** {plant}"
        )

        st.write(
            f"**AI Prediction:** {prediction}"
        )

        st.write(
            f"**Confidence:** "
            f"{confidence * 100:.2f}%"
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

        if confidence < 0.40:

            st.warning(
                "⚠️ The AI has low confidence. "
                "The result may not be reliable."
            )

        if plant in ["Guava", "Mango"]:

            st.error(
                f"⚠️ The current AI model was not trained "
                f"specifically on {plant} leaves. "
                f"This result should not be considered "
                f"a reliable {plant} disease diagnosis."
            )

        else:

            st.success(
                "Analysis complete!"
            )

else:

    st.info(
        "Upload a clear leaf image to begin."
    )


st.divider()

st.subheader(
    "📖 About the Project"
)

st.write(
    """
    The Plant Disease Detector is an AI-based smart
    agriculture project that analyzes plant leaf images
    and predicts possible diseases.

    Users should always consider the confidence score
    and verify uncertain results with an agricultural
    expert.
    """
)
