import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from datetime import datetime
import os


# PAGE SETTINGS
st.set_page_config(
    page_title="Smart Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)


# TITLE
st.title("🌿 Smart Plant Disease Detector")
st.write("AI-powered plant leaf disease identification system")


# MODEL LOADING
MODEL_PATH = "model_float16_quant.tflite"


@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


try:
    interpreter = load_model()
    st.success("AI Model Loaded Successfully")
except Exception as e:
    st.error("Model not found. Upload model_float16_quant.tflite")
    st.stop()


# CLASS LABELS
CLASS_NAMES = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Healthy",
    "Corn Leaf Blight",
    "Corn Healthy",
    "Grape Black Rot",
    "Grape Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy"
]


# RECOMMENDATIONS

def advice(result):

    if "Healthy" in result:
        return """
Your plant appears healthy.

Recommendations:
- Continue proper watering.
- Maintain sunlight exposure.
- Monitor regularly for changes.
"""

    elif "Blight" in result:
        return """
Possible blight disease detected.

Actions:
- Remove infected leaves.
- Avoid excessive watering.
- Improve air circulation.
- Apply recommended fungicide.
"""

    elif "Scab" in result:
        return """
Possible scab disease detected.

Actions:
- Remove affected leaves.
- Keep leaves dry.
- Use suitable fungicide.
- Monitor plant development.
"""

    else:
        return """
Possible fungal infection detected.

Actions:
- Separate infected plants.
- Remove damaged parts.
- Maintain good farming hygiene.
"""


# IMAGE PREPROCESSING

def prepare_image(image):

    img = image.resize((224,224))
    img = np.array(img)

    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img,axis=0)

    return img



# PREDICTION

def predict(image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    data = prepare_image(image)

    interpreter.set_tensor(
        input_details[0]['index'],
        data
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]['index']
    )

    index = np.argmax(prediction)

    confidence = float(np.max(prediction))*100

    if index < len(CLASS_NAMES):
        disease = CLASS_NAMES[index]
    else:
        disease = "Unknown"


    return disease, confidence



# UPLOAD IMAGE

uploaded = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg","jpeg","png"]
)


if uploaded:

    image = Image.open(uploaded)

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )


    if st.button("🔍 Analyse Plant"):

        with st.spinner("AI analysing image..."):

            disease, confidence = predict(image)


        st.subheader("🌿 AI Prediction")

        st.write(
            "Plant Disease:",
            disease
        )

        st.write(
            "Confidence:",
            f"{confidence:.2f}%"
        )

        st.write(
            "Analysis Time:",
            datetime.now().strftime("%H:%M:%S")
        )


        if confidence < 60:

            st.warning(
                "Low confidence. Try uploading a clearer leaf image."
            )


        st.subheader("🌱 Recommended Actions")

        st.info(
            advice(disease)
        )


        report = f"""
SMART PLANT DISEASE DETECTOR REPORT

Prediction:
{disease}

Confidence:
{confidence:.2f}%

Date:
{datetime.now()}
"""


        st.download_button(
            "Download Analysis Report",
            report,
            file_name="plant_report.txt"
        )


# ABOUT

st.divider()

st.subheader("About The Project")

st.write(
"""
Smart Plant Disease Detector uses Artificial Intelligence
and image recognition technology to identify possible
plant diseases from leaf images.

It helps farmers and students understand plant health
and provides recommended management practices.
"""
)
