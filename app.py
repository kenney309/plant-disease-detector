import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Smart Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# =========================================================
# BEAUTIFUL AGRICULTURE-THEMED DESIGN
# =========================================================

st.markdown(
    """
    <style>

    /* Main application background */
    .stApp {
        background:
        linear-gradient(
            rgba(240, 248, 240, 0.95),
            rgba(220, 240, 225, 0.95)
        );
    }

    /* Main title */
    h1 {
        color: #1B5E20;
        text-align: center;
        font-size: 42px;
        font-weight: bold;
    }

    /* Section headings */
    h2, h3 {
        color: #2E7D32;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        background-color: #2E7D32;
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background-color: #1B5E20;
        color: white;
    }

    /* Information boxes */
    .stAlert {
        border-radius: 12px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 15px;
        padding: 10px;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #43A047;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HEADER
# =========================================================

st.title("🌿 Smart Plant Disease Detector")

st.markdown(
    """
    <div style="
        text-align:center;
        padding:15px;
        border-radius:15px;
        background-color:rgba(255,255,255,0.7);
        margin-bottom:20px;
    ">

    <h3>🤖 AI for Smart Agriculture</h3>

    <p>
    Upload a clear image of a plant leaf and let
    Artificial Intelligence analyze it for possible
    plant diseases.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

st.divider()
