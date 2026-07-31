import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os
import time
from datetime import datetime


st.set_page_config(
    page_title="Smart Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)


# ---------- STYLE ----------
st.markdown("""
<style>

.main {
    background-color: #f4f8f4;
}

.title {
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:#1b5e20;
}

.subtitle {
    text-align:center;
    font-size:20px;
    color:#555;
}


.card {
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
    margin:10px;
}


.result {
    background:#e8f5e9;
    padding:20px;
    border-radius:15px;
    border-left:8px solid #2e7d32;
}


.info {
    background:#fffde7;
    padding:20px;
    border-radius:15px;
}


</style>
""", unsafe_allow_html=True)



# ---------- MODEL ----------
MODEL_URL = "https://huggingface.co/animeshakr/plant-disease-efficientnetv2s/resolve/main/model_float16_quant.tflite"

MODEL_PATH = "plant_model.tflite"



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

"Apple Scab":
"Remove infected leaves, improve airflow and apply recommended fungicide.",

"Apple Black Rot":
"Remove infected parts and maintain good orchard hygiene.",

"Corn Leaf Blight":
"Use resistant varieties and avoid excessive moisture.",

"Corn Common Rust":
"Monitor plants and apply suitable fungicide.",

"Grape Black Rot":
"Remove infected fruits and improve field sanitation.",

"Potato Early Blight":
"Practice crop rotation and remove affected leaves.",

"Potato Late Blight":
"Use approved fungicides and avoid wet leaves.",

"Tomato Early Blight":
"Remove infected leaves and improve spacing.",

"Tomato Late Blight":
"Remove affected plants and apply treatment.",

"Healthy":
"Plant appears healthy. Continue proper farming practices."

}



# ---------- DOWNLOAD MODEL ----------
def download_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Preparing AI model..."):

            r = requests.get(MODEL_URL)

            with open(MODEL_PATH,"wb") as file:
                file.write(r.content)

    return MODEL_PATH




@st.cache_resource
def load_model():

    model = download_model()

    interpreter = tf.lite.Interpreter(
        model_path=model
    )

    interpreter.allocate_tensors()

    return interpreter




def predict(image):

    interpreter = load_model()


    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()


    image = image.resize((224,224))


    img = np.array(image)


    img = img.astype(np.float32)/255.0


    img = np.expand_dims(img,axis=0)


    interpreter.set_tensor(
        input_details[0]['index'],
        img
    )


    interpreter.invoke()


    output = interpreter.get_tensor(
        output_details[0]['index']
    )


    index = np.argmax(output)


    confidence = float(np.max(output))*100


    if index < len(CLASS_NAMES):

        disease = CLASS_NAMES[index]

    else:

        disease="Unknown"



    return disease,confidence




# ---------- HEADER ----------

st.markdown(
"""
<div class="title">
🌿 Smart Plant Disease Detector
</div>

<div class="subtitle">
AI-powered crop disease identification system
</div>
""",
unsafe_allow_html=True
)


st.write("")



# ---------- UPLOAD ----------

col1,col2 = st.columns(2)



with col1:

    st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
    )

    st.subheader("📸 Upload Plant Leaf")

    uploaded = st.file_uploader(
        "Choose leaf image",
        type=["jpg","jpeg","png"]
    )

    st.markdown("</div>",unsafe_allow_html=True)




with col2:

    st.markdown(
    """
    <div class="info">

    🌱 <b>How it works</b>

    <br><br>

    1. Upload a clear leaf image

    <br>
    2. AI analyses the leaf

    <br>
    3. Disease prediction is displayed

    <br>
    4. Receive farming advice

    </div>
    """,
    unsafe_allow_html=True
    )



if uploaded:


    image = Image.open(uploaded)


    st.image(
        image,
        caption="Uploaded Leaf",
        width=350
    )


    if st.button("🔍 Analyse Plant"):


        start=time.time()


        with st.spinner("AI analysing..."):

            disease,confidence = predict(image)



        total_time=round(
            time.time()-start,
            2
        )



        st.markdown(
        '<div class="result">',
        unsafe_allow_html=True
        )


        st.subheader("🌿 AI Prediction")


        st.write(
            "Plant Condition:",
            disease
        )


        st.write(
            "Confidence:",
            f"{confidence:.2f}%"
        )


        st.write(
            "Analysis Time:",
            str(total_time)+" seconds"
        )


        st.write(
            "Date:",
            datetime.now().strftime("%d-%m-%Y")
        )


        st.markdown(
        "</div>",
        unsafe_allow_html=True
        )



        if confidence < 60:

            st.warning(
            "⚠ Low confidence. Upload a clearer leaf image."
            )


        st.subheader("💡 Recommended Action")


        st.info(
            RECOMMENDATIONS.get(
                disease,
                "Consult an agricultural expert."
            )
        )



# ---------- ABOUT ----------

st.divider()


st.subheader("ℹ️ About The Project")


st.write(
"""
Smart Plant Disease Detector uses Artificial Intelligence
and image recognition to identify possible crop diseases
from leaf images.

The system helps students, farmers and agricultural workers
make faster decisions about plant health.
"""
)
