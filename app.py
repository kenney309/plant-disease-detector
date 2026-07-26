import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detector")

st.write(
    "Upload a clear image of a plant leaf "
    "to detect possible diseases using Artificial Intelligence."
)

st.divider()

uploaded_file = st.file_uploader(
    "📷 Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Plant Leaf",
        use_container_width=True
    )

    if st.button("🔍 Analyze Leaf"):

        st.success("Image received successfully!")

        st.subheader("🌿 Analysis Result")

        st.write("**Plant:** Image uploaded")

        st.write(
            "**Status:** Ready for AI disease classification"
        )

        st.info(
            "The plant disease AI model will analyze this "
            "image in the next stage."
        )

else:

    st.info(
        "👆 Upload a clear picture of a plant leaf to begin."
    )

st.divider()

st.subheader("📖 About the Project")

st.write("""
The Plant Disease Detector is an Artificial Intelligence
project designed to support smart agriculture.

The system is designed to analyze images of plant leaves
and identify possible diseases, helping users make better
decisions about plant health.
""")

st.caption(
    "🌿 Plant Disease Detector | AI for Smart Agriculture"
)
