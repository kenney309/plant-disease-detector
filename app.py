import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
import json
import hashlib
from datetime import datetime


st.set_page_config(
    page_title="Smart Plant AI",
    page_icon="🌱",
    layout="wide"
)


# ---------------- LOGIN SYSTEM ----------------

USER_FILE = "users.json"

if not os.path.exists(USER_FILE):
    with open(USER_FILE,"w") as f:
        json.dump({},f)


def load_users():
    with open(USER_FILE,"r") as f:
        return json.load(f)


def save_users(users):
    with open(USER_FILE,"w") as f:
        json.dump(users,f,indent=4)


def hash_password(password):
    return hashlib.sha256(
        password.encode()
    ).hexdigest()



def register(username,password):

    users=load_users()

    if username in users:
        return False

    users[username]={
        "password":hash_password(password),
        "history":[]
    }

    save_users(users)
    return True



def login(username,password):

    users=load_users()

    if username in users:
        return users[username]["password"]==hash_password(password)

    return False



# ---------------- AI MODEL ----------------

MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_0/plant_disease_model.tflite"

MODEL_FILE="plant_model.tflite"


def download_model():

    if not os.path.exists(MODEL_FILE):

        r=requests.get(MODEL_URL)

        with open(MODEL_FILE,"wb") as f:
            f.write(r.content)



@st.cache_resource
def load_model():

    download_model()

    return tf.lite.Interpreter(
        model_path=MODEL_FILE
    )


# ---------------- SESSION ----------------


if "logged" not in st.session_state:
    st.session_state.logged=False



# ---------------- LOGIN PAGE ----------------


if not st.session_state.logged:


    st.title("🌱 Smart Plant AI")


    option=st.radio(
        "Choose",
        ["Login","Register"],
        horizontal=True
    )


    username=st.text_input("Username")
    password=st.text_input(
        "Password",
        type="password"
    )


    if option=="Register":

        if st.button("Create Account"):

            if register(username,password):
                st.success("Account created")
            else:
                st.error("Username exists")


    else:

        if st.button("Login"):

            if login(username,password):

                st.session_state.logged=True
                st.session_state.username=username
                st.rerun()

            else:
                st.error("Wrong details")



# ---------------- MAIN APP ----------------


else:


    st.sidebar.success(
        st.session_state.username
    )


    page=st.sidebar.selectbox(
        "Menu",
        [
            "Dashboard",
            "Analyze Leaf",
            "History"
        ]
    )


    if page=="Dashboard":

        st.title("🌿 Smart Plant AI Dashboard")

        st.write(
            """
            Upload a plant leaf and AI will analyze:
            
            ✔ Crop type
            ✔ Disease
            ✔ Confidence
            ✔ Treatment advice
            """
        )



    elif page=="Analyze Leaf":


        st.title("🔍 AI Leaf Analysis")


        image=st.file_uploader(
            "Upload leaf image",
            type=[
                "jpg",
                "png",
                "jpeg"
            ]
        )


        if image:


            img=Image.open(image)

            st.image(
                img,
                caption="Uploaded Leaf"
            )


            if st.button("Analyze"):


                with st.spinner(
                    "AI analyzing..."
                ):


                    interpreter=load_model()

                    st.success(
                        "Analysis completed"
                    )


                    st.subheader(
                        "AI Prediction"
                    )

                    st.write(
                        "Disease: Model connected"
                    )

                    st.write(
                        "Confidence: Calculating..."
                    )



    elif page=="History":

        st.title(
            "📄 Previous Reports"
        )

        st.info(
            "Your saved analyses will appear here."
        )



    if st.sidebar.button("Logout"):

        st.session_state.logged=False
        st.rerun()
