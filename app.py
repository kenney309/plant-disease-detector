import streamlit as st

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿"
)

st.title("🌿 PLANT DISEASE DETECTOR")

st.write("WELCOME TO MY PLANT DISEASE DETECTOR")

st.header("📷 Upload a Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Your Uploaded Leaf"
    )

    st.success("Leaf image uploaded successfully!")

    if st.button("🔍 ANALYZE LEAF"):
        st.success("ANALYSIS BUTTON WORKS!")

st.header("📖 About the Project")

st.write(
    "This project uses Artificial Intelligence to help "
    "identify possible plant diseases from leaf images."
)
