import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

# =========================================================
# CUSTOM HEADER
# =========================================================

st.title("🌿 Smart Plant Disease Detector")

st.markdown(
    """
    ### AI-Powered Smart Agriculture System

    Upload or capture a clear image of a plant leaf.
    The system will analyze the image and provide a possible
    disease prediction, confidence score, and general guidance.
    """
)

st.divider()

# =========================================================
# MODEL
# =========================================================

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

# =========================================================
# DISEASE INFORMATION
# =========================================================

DISEASE_INFO = {

    "Apple - Apple Scab": {
        "symptoms": "Dark or olive-colored spots may appear on leaves and fruit.",
        "treatment": "Remove heavily affected plant material and consider appropriate fungicide management.",
        "prevention": "Improve air circulation and remove fallen infected leaves."
    },

    "Grape - Black Rot": {
        "symptoms": "Dark spots and lesions may develop on leaves and fruit.",
        "treatment": "Remove infected material and use appropriate disease management practices.",
        "prevention": "Maintain good vineyard sanitation and air circulation."
    },

    "Grape - Esca (Black Measles)": {
        "symptoms": "Leaves may develop irregular discoloration and the plant may weaken.",
        "treatment": "Remove severely affected plant parts and seek expert agricultural advice.",
        "prevention": "Use healthy planting material and maintain good plant hygiene."
    },

    "Grape - Leaf Blight": {
        "symptoms": "Brown or dark lesions can appear on leaves.",
        "treatment": "Remove infected leaves and use suitable disease management methods.",
        "prevention": "Avoid excessive leaf wetness and improve air circulation."
    },

    "Tomato - Early Blight": {
        "symptoms": "Brown spots with ring-like patterns may develop on older leaves.",
        "treatment": "Remove affected leaves and use appropriate fungicide management when recommended.",
        "prevention": "Practice crop rotation and avoid watering foliage unnecessarily."
    },

    "Tomato - Late Blight": {
        "symptoms": "Dark irregular lesions can develop rapidly on leaves and stems.",
        "treatment": "Remove severely infected material and seek agricultural advice.",
        "prevention": "Improve air circulation and avoid prolonged leaf wetness."
    },

    "Tomato - Healthy": {
        "symptoms": "No major disease symptoms were detected.",
        "treatment": "Continue normal plant care and monitor the plant regularly.",
        "prevention": "Maintain good sanitation, nutrition, and watering practices."
    },

    "Apple - Healthy": {
        "symptoms": "No major disease symptoms were detected.",
        "treatment": "Continue normal plant care.",
        "prevention": "Monitor the plant regularly and maintain good sanitation."
    },

    "Grape - Healthy": {
        "symptoms": "No major disease symptoms were detected.",
        "treatment": "Continue normal plant care.",
        "prevention": "Maintain good air circulation and monitor regularly."
    }
}

# =========================================================
# LOAD MODEL
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

                file.write(response.content)

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
# PREDICTION FUNCTION
# =========================================================

def predict_image(image):

    interpreter, inputs, outputs = load_model()

    input_shape = inputs[0]["shape"]

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
        inputs[0]["index"],
        image_array
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        outputs[0]["index"]
    )[0]

    top_indices = np.argsort(
        prediction
    )[::-1][:3]

    results = []

    for index in top_indices:

        if index < len(CLASS_NAMES):

            name = CLASS_NAMES[index]

        else:

            name = "Unknown"

        confidence = float(
            prediction[index]
        )

        results.append(
            (name, confidence)
        )

    return results

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌱 Plant Information")

plant = st.sidebar.selectbox(
    "Select the plant you are testing:",
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

st.sidebar.info(
    "The current AI model supports a specific set of "
    "plant and disease categories."
)

# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("📷 Upload or Capture a Leaf Image")

input_method = st.radio(
    "Choose image source:",
    [
        "Upload Image",
        "Use Camera"
    ],
    horizontal=True
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
        "Take a picture of the leaf"
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
        caption=f"{plant} Leaf Image",
        use_container_width=True
    )

    if plant in ["Guava", "Mango", "Banana"]:

        st.warning(
            f"⚠️ Note: The current AI model was not "
            f"specifically trained to recognize {plant}. "
            f"Predictions for this plant may be inaccurate."
        )

    if st.button(
        "🔍 ANALYZE LEAF",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "🤖 AI is analyzing the leaf..."
            ):

                results = predict_image(
                    image
                )

            best_name = results[0][0]

            best_confidence = results[0][1]

            st.success(
                "Analysis complete!"
            )

            st.divider()

            # =================================================
            # MAIN RESULT
            # =================================================

            st.subheader(
                "🌿 Main AI Prediction"
            )

            st.write(
                f"**Selected Plant:** {plant}"
            )

            st.write(
                f"**AI Prediction:** {best_name}"
            )

            st.write(
                f"**Confidence:** "
                f"{best_confidence * 100:.2f}%"
            )

            st.progress(
                min(
                    max(
                        best_confidence,
                        0.0
                    ),
                    1.0
                )
            )

            # =================================================
            # CONFIDENCE WARNING
            # =================================================

            if best_confidence < 0.40:

                st.error(
                    "⚠️ LOW CONFIDENCE RESULT: "
                    "The AI is not confident about this prediction. "
                    "The plant may not be supported by the model "
                    "or the image may be unclear."
                )

            elif best_confidence < 0.70:

                st.warning(
                    "⚠️ MODERATE CONFIDENCE: "
                    "Consider verifying this result."
                )

            else:

                st.success(
                    "✅ HIGHER CONFIDENCE RESULT"
                )

            # =================================================
            # TOP 3
            # =================================================

            st.subheader(
                "🔎 Top 3 AI Predictions"
            )

            for number, (
                name,
                score
            ) in enumerate(
                results,
                start=1
            ):

                st.write(
                    f"**{number}. {name}** — "
                    f"{score * 100:.2f}%"
                )

            # =================================================
            # DISEASE INFORMATION
            # =================================================

            if best_name in DISEASE_INFO:

                info = DISEASE_INFO[
                    best_name
                ]

                st.divider()

                st.subheader(
                    "📚 Disease Information"
                )

                st.write(
                    f"**Symptoms:** "
                    f"{info['symptoms']}"
                )

                st.write(
                    f"**General Management:** "
                    f"{info['treatment']}"
                )

                st.write(
                    f"**Prevention:** "
                    f"{info['prevention']}"
                )

            else:

                st.info(
                    "Detailed information for this prediction "
                    "is not yet available in the app."
                )

            # =================================================
            # GENERAL ADVICE
            # =================================================

            st.divider()

            st.subheader(
                "👨‍🌾 General Farmer Advice"
            )

            st.write(
                """
                • Keep infected leaves away from healthy plants.

                • Maintain good field and garden sanitation.

                • Monitor plants regularly for changes.

                • Avoid unnecessary leaf wetness.

                • Use appropriate agricultural treatments
                  according to local expert advice.

                • For serious or uncertain cases, consult
                  an agricultural extension officer.
                """
            )

        except Exception as e:

            st.error(
                "❌ An error occurred during analysis."
            )

            st.write(
                str(e)
            )

else:

    st.info(
        "📷 Upload or capture a clear leaf image "
        "to begin analysis."
    )

# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.subheader(
    "📖 About This Project"
)

st.write(
    """
    The Smart Plant Disease Detector is an Artificial
    Intelligence project designed to support smart agriculture.

    The system analyzes plant leaf images and provides possible
    disease predictions, confidence scores, and general
    agricultural guidance.

    The system is intended as a supporting tool and should not
    replace professional agricultural diagnosis.
    """
)

st.caption(
    "🌿 Smart Plant Disease Detector | AI for Smart Agriculture"
)
