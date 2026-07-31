import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from datetime import datetime
import os


st.set_page_config(
    page_title="Smart Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)


st.title("🌿 Smart Plant Disease Detector")
st.write("AI system for detecting plant diseases from leaf images")


MODEL_FILE = "model_float16_quant.tflite"


@st.cache_resource
def load_model():

    interpreter = tf.lite.Interpreter(
        model_path=MODEL_FILE
    )

    interpreter.allocate_tensors()

    return interpreter



if not os.path.exists(MODEL_FILE):

    st.error(
        "Model file missing. Upload model_float16_quant.tflite"
    )

    st.stop()



interpreter = load_model()



classes = [
"Apple Scab",
"Apple Black Rot",
"Apple Healthy",
"Corn Blight",
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



def preprocess(img):

    img = img.resize((224,224))

    img = np.array(img)

    img = img.astype(np.float32)

    img = img / 255.0

    img = np.expand_dims(img,0)

    return img



def prediction(image):

    input_data = interpreter.get_input_details()

    output_data = interpreter.get_output_details()


    processed = preprocess(image)


    interpreter.set_tensor(
        input_data[0]["index"],
        processed
    )


    interpreter.invoke()


    output = interpreter.get_tensor(
        output_data[0]["index"]
    )


    result = np.argmax(output)

    confidence = np.max(output)*100


    if result < len(classes):

        return classes[result], confidence

    else:

        return "Unknown", confidence




def recommendations(disease):

    if "Healthy" in disease:

        return """
✅ Plant looks healthy.

Maintain:
• Proper watering
• Good sunlight
• Regular monitoring
"""


    if "Blight" in disease:

        return """
⚠ Possible Blight disease.

Actions:
• Remove infected leaves
• Improve air circulation
• Avoid overwatering
• Apply suitable treatment
"""


    if "Scab" in disease:

        return """
⚠ Possible Apple Scab.

Actions:
• Remove affected leaves
• Keep leaves dry
• Use recommended fungicide
"""


    return """
⚠ Possible disease detected.

Actions:
• Isolate affected plant
• Remove damaged parts
• Monitor progress
"""




uploaded = st.file_uploader(
    "Upload plant leaf image",
    type=["png","jpg","jpeg"]
)



if uploaded:


    img = Image.open(uploaded)


    st.image(
        img,
        caption="Uploaded Leaf",
        width=400
    )



    if st.button("🔍 Detect Disease"):


        disease, confidence = prediction(img)



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
            datetime.now()
        )



        if confidence < 50:

            st.warning(
            "Low confidence. Upload a clearer leaf image."
            )


        st.subheader("🌱 Recommendations")


        st.info(
            recommendations(disease)
        )



        report = f"""
SMART PLANT DISEASE DETECTOR REPORT

Disease:
{disease}

Confidence:
{confidence:.2f}%

Time:
{datetime.now()}
"""


        st.download_button(
            "Download Report",
            report,
            file_name="plant_report.txt"
        )



st.divider()


st.subheader("About The Project")

st.write(
"""
Smart Plant Disease Detector uses Artificial Intelligence
to analyse plant leaves and identify possible diseases.
It provides farmers and students with quick diagnosis
and recommended management practices.
"""
)
