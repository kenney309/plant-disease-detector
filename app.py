import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detector")

st.write(
    "An AI-powered system designed to help identify "
    "possible plant diseases from leaf images."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a clear picture of a plant leaf",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Plant Leaf",
        use_container_width=True
    )

    st.success("Image uploaded successfully!")

    if st.button("🔍 Analyze Leaf"):

        st.info(
            "The AI model is ready to analyze this image. "
            "The plant disease classification model will be "
            "connected next."
        )

else:

    st.info(
        "Upload a clear image of a plant leaf to begin."
    )

st.divider()

st.subheader("📖 About This Project")

st.write("""
The Plant Disease Detector is an Artificial Intelligence
project designed to support smart agriculture.

The system analyzes images of plant leaves and identifies
possible diseases. It can help farmers and agricultural
communities make better decisions about plant health.
""")
