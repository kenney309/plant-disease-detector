import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

# ============================================
# PAGE SETTINGS
# ============================================

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detector")

st.write(
    "Upload a clear image of a plant leaf and the AI "
    "will analyze it for possible plant diseases."
)

st.divider()


# ============================================
# MODEL SETTINGS
# ============================================

MODEL_URL = (
    "https://huggingface.co/animeshakr/"
    "plant-disease-efficientnetv2s/resolve/main/"
    "model_float16_quant.tflite"
)

MODEL_PATH = "plant_disease_model.tflite"


# ============================================
# CLASS NAMES
# ============================================

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


# ============================================
# DOWNLOAD MODEL
# ============================================

@st.cache_resource
def download_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner(
            "Downloading AI model for the first time..."
        ):

            response = requests.get(
                MODEL_URL,
                stream=True,
                timeout=300
            )

            response.raise_for_status()

            with open(
                MODEL_PATH,
                "wb"
            ) as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        file.write(chunk)

    return MODEL_PATH


# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():

    model_path = download_model()

    interpreter = tf.lite.Interpreter(
        model_path=model_path
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    return (
        interpreter,
        input_details,
        output_details
    )


# ============================================
# PREDICT IMAGE
# ============================================

def predict_image(image):

    interpreter, input_details, output_details = load_model()

    input_shape = input_details[0]["shape"]

    height = input_shape[1]
    width = input_shape[2]

    # Resize image
    image = image.resize(
        (width, height)
    )

    # Convert to RGB
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize image
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Run AI
    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    predictions = output[0]

    # Get top 3 predictions
    top_indices = np.argsort(
        predictions
    )[::-1][:3]

    results = []

    for index in top_indices:

        if index < len(CLASS_NAMES):

            class_name = CLASS_NAMES[index]

        else:

            class_name = (
                f"Unknown class {index}"
            )

        confidence = float(
            predictions[index]
        )

        results.append(
            (
                class_name,
                confidence
            )
        )

    return results


# ============================================
# UPLOAD IMAGE
# ============================================

uploaded_file = st.file_uploader(
    "📷 Upload a plant leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================
# ANALYZE IMAGE
# ============================================

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
        "🔍 ANALYZE LEAF"
    ):

        try:

            with st.spinner(
                "🤖 AI is analyzing the leaf..."
            ):

                results = predict_image(
                    image
                )

            # Best prediction
            best_prediction = results[0]

            disease = best_prediction[0]

            confidence = best_prediction[1]

            st.success(
                "Analysis complete!"
            )

            st.subheader(
                "🌿 AI Analysis"
            )

            # ====================================
            # LOW CONFIDENCE WARNING
            # ====================================

            if confidence < 0.40:

                st.warning(
                    "⚠️ LOW CONFIDENCE: "
                    "The AI is not confident about this result. "
                    "The plant may not be supported by the current "
                    "AI model, or the image may be unclear."
                )

            else:

                st.success(
                    "The AI has a higher confidence "
                    "in this prediction."
                )


            # ====================================
            # MAIN PREDICTION
            # ====================================

            st.write(
                f"### Prediction: {disease}"
            )

            st.write(
                f"**Confidence: "
                f"{confidence * 100:.2f}%**"
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


            # ====================================
            # TOP 3 PREDICTIONS
            # ====================================

            st.subheader(
                "🔎 Top 3 AI Predictions"
            )

            for i, (
                class_name,
                score
            ) in enumerate(
                results,
                start=1
            ):

                st.write(
                    f"**{i}. {class_name}** — "
                    f"{score * 100:.2f}%"
                )


            # ====================================
            # IMPORTANT INFORMATION
            # ====================================

            st.info(
                "💡 Important: This AI model was trained "
                "on a specific set of plant and disease "
                "categories. If your plant is not one of "
                "the supported categories, the AI may "
                "incorrectly choose another plant."
            )


        except Exception as e:

            st.error(
                "❌ An error occurred while analyzing "
                "the image."
            )

            st.write(
                str(e)
            )


else:

    st.info(
        "📷 Upload a clear plant leaf image "
        "to begin."
    )


# ============================================
# ABOUT PROJECT
# ============================================

st.divider()

st.subheader(
    "📖 About the Project"
)

st.write(
    """
    The Plant Disease Detector is an Artificial Intelligence
    project designed to support smart agriculture.

    The system analyzes images of plant leaves and predicts
    possible plant disease categories.

    The system also provides a confidence score and warns
    users when the AI is not confident about its prediction.
    """
)

st.caption(
    "🌿 Plant Disease Detector | AI for Smart Agriculture"
)
