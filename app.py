import streamlit as st
from PIL import Image
import requests

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detector")

st.write(
    "Upload a clear image of a plant leaf to identify "
    "possible plant diseases using Artificial Intelligence."
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

    if st.button("🔍 ANALYZE LEAF"):

        with st.spinner("AI is analyzing the leaf..."):

            try:

                # Get Hugging Face token from Streamlit Secrets
                hf_token = st.secrets["HF_TOKEN"]

                # Hugging Face model endpoint
                API_URL = (
                    "https://api-inference.huggingface.co/models/"
                    "animeshakr/plant-disease-efficientnetv2s"
                )

                headers = {
                    "Authorization": f"Bearer {hf_token}"
                }

                # Convert image to bytes
                image_bytes = uploaded_file.getvalue()

                # Send image to model
                response = requests.post(
                    API_URL,
                    headers=headers,
                    data=image_bytes,
                    timeout=60
                )

                if response.status_code == 200:

                    results = response.json()

                    st.success("Analysis complete!")

                    st.subheader("🌿 AI Prediction")

                    if isinstance(results, list):

                        best_result = max(
                            results,
                            key=lambda x: x.get(
                                "score", 0
                            )
                        )

                        label = best_result.get(
                            "label",
                            "Unknown"
                        )

                        score = best_result.get(
                            "score",
                            0
                        )

                        st.write(
                            f"**Prediction:** {label}"
                        )

                        st.write(
                            f"**Confidence:** "
                            f"{score * 100:.2f}%"
                        )

                        st.progress(
                            min(float(score), 1.0)
                        )

                    else:

                        st.write(results)

                else:

                    st.error(
                        "The AI model could not process "
                        "the image."
                    )

                    st.write(
                        f"Server response: "
                        f"{response.status_code}"
                    )

                    st.write(response.text)

            except Exception as e:

                st.error(
                    "An error occurred while analyzing "
                    "the image."
                )

                st.write(str(e))

else:

    st.info(
        "👆 Upload a clear plant leaf image "
        "to begin."
    )

st.divider()

st.subheader("📖 About the Project")

st.write(
    """
    The Plant Disease Detector is an Artificial Intelligence
    project designed to support smart agriculture.

    The system analyzes images of plant leaves and predicts
    possible plant disease categories.
    """
)

st.caption(
    "🌿 Plant Disease Detector | AI for Smart Agriculture"
)
