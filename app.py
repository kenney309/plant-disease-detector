import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿"
)

st.title("🌿 Plant Disease Detector")

st.write(
    "Upload a picture of a plant leaf "
    "to begin disease detection."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )

    st.success("Image uploaded successfully!")

    if st.button("Analyze Leaf"):
        st.info(
            "The image is ready for AI analysis."
        )
