import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detector")
st.write(
    "Upload a clear image of a plant leaf and our AI "
    "will analyze it for possible diseases."
)

st.divider()

# -----------------------------
# MODEL SETTINGS
# -----------------------------
MODEL_URL = (
    "https://huggingface.co/animeshakr/"
    "plant-disease-efficientnetv2s/resolve/main/"
    "model_float16_quant.tflite"
)

MODEL_PATH = "plant_disease_model.tflite"

# 38 PlantVillage classes
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


# -----------------------------
# DISEASE INFORMATION
# -----------------------------
DISEASE_INFO = {

    "Apple - Apple Scab": {
        "description": "A fungal disease that affects apple leaves and fruit.",
        "symptoms": "Olive-green to dark spots may appear on leaves and fruit.",
        "advice": "Remove infected plant material where practical and improve air circulation."
    },

    "Apple - Black Rot": {
        "description": "A fungal disease that can affect apple leaves, fruit, and branches.",
        "symptoms": "Dark lesions and areas of tissue damage may develop.",
        "advice": "Remove affected plant material and maintain good orchard hygiene."
    },

    "Apple - Cedar Apple Rust": {
        "description": "A fungal disease affecting apple trees.",
        "symptoms": "Yellow or orange spots may develop on leaves.",
        "advice": "Remove severely affected leaves and seek local agricultural advice."
    },

    "Corn - Common Rust": {
        "description": "A fungal disease that affects maize leaves.",
        "symptoms": "Small rust-colored spots may appear on the leaves.",
        "advice": "Monitor the crop closely and seek agricultural advice for appropriate management."
    },

    "Corn - Northern Leaf Blight": {
        "description": "A fungal disease that can reduce the health of maize leaves.",
        "symptoms": "Long, grayish-green or brown lesions may appear on leaves.",
        "advice": "Remove heavily affected material where practical and seek professional agricultural advice."
    },

    "Grape - Black Rot": {
        "description": "A fungal disease affecting grape plants.",
        "symptoms": "Dark spots can develop on leaves and fruit.",
        "advice": "Remove affected plant material and maintain good vineyard hygiene."
    },

    "Grape - Leaf Blight": {
        "description": "A disease that causes damage to grape leaves.",
        "symptoms": "Discolored or damaged areas may develop on leaves.",
        "advice": "Remove severely affected leaves and improve plant care and air circulation."
    },

    "Peach - Bacterial Spot": {
        "description": "A bacterial disease that affects peach leaves and fruit.",
        "symptoms": "Small dark spots or lesions may appear on leaves and fruit.",
        "advice": "Remove badly affected material where practical and consult an agricultural professional."
    },

    "Pepper - Bacterial Spot": {
        "description": "A bacterial disease that affects pepper plants.",
        "symptoms": "Small dark spots or lesions may appear on leaves.",
        "advice": "Maintain good plant hygiene and seek professional agricultural advice."
    },

    "Potato - Early Blight": {
        "description": "A fungal disease commonly affecting potato leaves.",
        "symptoms": "Dark spots and rings may develop on older leaves.",
        "advice": "Remove severely affected leaves and maintain good field hygiene."
    },

    "Potato - Late Blight": {
        "description": "A serious disease that can rapidly affect potato plants.",
        "symptoms": "Dark lesions may develop on leaves and stems.",
        "advice": "Remove severely affected plant material where appropriate and seek professional advice."
    },

    "Squash - Powdery Mildew": {
        "description": "A fungal disease that commonly affects squash leaves.",
        "symptoms": "White powder-like patches may appear on leaves.",
        "advice": "Improve air circulation and remove severely affected leaves."
    },

    "Strawberry - Leaf Scorch": {
        "description": "A disease that causes damage to strawberry leaves.",
        "symptoms": "Dark or scorched-looking areas may appear on leaves.",
        "advice": "Remove severely affected leaves and maintain good plant hygiene."
    },

    "Tomato - Bacterial Spot": {
        "description": "A bacterial disease that affects tomato leaves and fruit.",
        "symptoms": "Small dark spots or lesions may develop on leaves and fruit.",
        "advice": "Remove severely affected material where practical and avoid handling plants when they are wet."
    },

    "Tomato - Early Blight": {
        "description": "A common fungal disease affecting tomato plants.",
        "symptoms": "Dark spots with concentric rings may appear on older leaves.",
        "advice": "Remove affected leaves where practical, maintain good field hygiene, and avoid wetting leaves during watering."
    },

    "Tomato - Late Blight": {
        "description": "A serious disease that can quickly damage tomato plants.",
        "symptoms": "Dark lesions may develop on leaves and other plant parts.",
        "advice": "Remove severely affected material where appropriate and seek advice from a qualified agricultural professional."
    },

    "Tomato - Leaf Mold": {
        "description": "A fungal disease that mainly affects tomato leaves.",
        "symptoms": "Yellowish areas may appear on the upper leaf surface with mold-like growth underneath.",
        "advice": "Improve air circulation and reduce excessive moisture around the leaves."
    },

    "Tomato - Septoria Leaf Spot": {
        "description": "A fungal disease that causes spots on tomato leaves.",
        "symptoms": "Small circular spots with darker edges may appear on leaves.",
        "advice": "Remove affected leaves where practical and maintain good garden hygiene."
    },

    "Tomato - Spider Mites": {
        "description": "Spider mites are tiny pests that feed on plant tissues.",
        "symptoms": "Leaves may show yellowing, speckling, or general damage.",
        "advice": "Monitor the plant closely and seek advice on appropriate pest management."
    },

    "Tomato - Target Spot": {
        "description": "A fungal disease affecting tomato leaves and fruit.",
        "symptoms": "Circular spots with target-like patterns may develop.",
        "advice": "Remove severely affected material and improve plant hygiene."
    },

    "Tomato - Yellow Leaf Curl Virus": {
        "description": "A viral disease that can severely affect tomato growth.",
        "symptoms": "Leaves may curl, become yellow, and show reduced growth.",
        "advice": "Remove severely affected plants where appropriate and consult a qualified agricultural professional."
    },

    "Tomato - Mosaic Virus": {
        "description": "A viral disease that affects tomato plants.",
        "symptoms": "Leaves may show mottled or mosaic-like patterns and reduced growth.",
        "advice": "Remove severely affected plants where appropriate and seek professional agricultural advice."
    }
}


# -----------------------------
# DOWNLOAD MODEL
# -----------------------------
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

            with open(MODEL_PATH, "wb") as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        file.write(chunk)

    return MODEL_PATH


# -----------------------------
# LOAD MODEL
# -----------------------------
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


# -----------------------------
# PREDICTION
# -----------------------------
def predict_image(image):

    interpreter, input_details, output_details = load_model()

    input_shape = input_details[0]["shape"]

    height = input_shape[1]
    width = input_shape[2]

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

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

    else:

        predicted_class = "Unknown"

    return (
        predicted_class,
        confidence
    )


# -----------------------------
# UPLOAD IMAGE
# -----------------------------
uploaded_file = st.file_uploader(
    "📷 Upload a plant leaf image",
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
        "🔍 ANALYZE LEAF"
    ):

        try:

            with st.spinner(
                "🤖 AI is analyzing the leaf..."
            ):

                disease, confidence = predict_image(
                    image
                )

            # -----------------------------
            # RESULT
            # -----------------------------
            st.success(
                "Analysis complete!"
            )

            st.divider()

            st.subheader(
                "🌿 AI Analysis Result"
            )

            # Extract plant name
            if " - " in disease:

                plant_name, disease_name = disease.split(
                    " - ",
                    1
                )

            else:

                plant_name = "Unknown"
                disease_name = disease

            # Healthy or diseased
            if "Healthy" in disease:

                status = "🟢 HEALTHY"

            else:

                status = "🔴 POSSIBLE DISEASE DETECTED"

            st.markdown(
                f"### {status}"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "🌱 Plant",
                    plant_name
                )

            with col2:

                st.metric(
                    "📊 Confidence",
                    f"{confidence * 100:.2f}%"
                )

            st.write(
                f"**🦠 Disease:** {disease_name}"
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

            # -----------------------------
            # DISEASE INFORMATION
            # -----------------------------
            if disease in DISEASE_INFO:

                info = DISEASE_INFO[
                    disease
                ]

                st.subheader(
                    "📖 About This Condition"
                )

                st.write(
                    info["description"]
                )

                st.subheader(
                    "🔎 Common Symptoms"
                )

                st.write(
                    info["symptoms"]
                )

                st.subheader(
                    "💡 Management Advice"
                )

                st.info(
                    info["advice"]
                )

            elif "Healthy" in disease:

                st.subheader(
                    "🌱 Plant Health Advice"
                )

                st.success(
                    "The AI model classified this leaf as "
                    "healthy. Continue proper watering, "
                    "adequate sunlight, good nutrition, and "
                    "regular monitoring."
                )

            else:

                st.subheader(
                    "💡 Management Advice"
                )

                st.info(
                    "The AI detected a possible plant health "
                    "problem. Monitor the plant closely and "
                    "consult a qualified agricultural "
                    "professional for a specific diagnosis."
                )

            st.warning(
                "⚠️ This AI result is for educational and "
                "screening purposes. It should not replace "
                "professional agricultural diagnosis."
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
        "👆 Upload a clear plant leaf image to begin."
    )


# -----------------------------
# ABOUT PROJECT
# -----------------------------
st.divider()

st.subheader(
    "📖 About the Project"
)

st.write(
    """
    The Plant Disease Detector is an Artificial Intelligence
    project designed to support smart agriculture.

    The system analyzes images of plant leaves and predicts
    possible plant disease categories. It provides the plant
    name, disease prediction, confidence score, symptoms,
    and basic management information.
    """
)

st.caption(
    "🌿 Plant Disease Detector | AI for Smart Agriculture"
)
