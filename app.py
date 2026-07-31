import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import hashlib
import os
from datetime import datetime


st.set_page_config(
    page_title="Smart Plant AI",
    page_icon="🌱",
    layout="wide"
)


# ---------------- USER DATABASE ----------------

USER_FILE = "users.json"


if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)


def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)


def hash_password(password):
    return hashlib.sha256(
        password.encode()
    ).hexdigest()



def register(username, password):

    users = load_users()

    if username in users:
        return False

    users[username] = {
        "password": hash_password(password),
        "history": []
    }

    save_users(users)

    return True



def check_login(username,password):

    users = load_users()

    if username in users:

        return users[username]["password"] == hash_password(password)

    return False



# ---------------- AI MODEL ----------------


MODEL_PATH = "model.tflite"
LABEL_PATH = "labels.txt"


@st.cache_resource
def load_ai():

    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    return interpreter



def load_labels():

    with open(LABEL_PATH,"r") as f:
        return [
            line.strip()
            for line in f.readlines()
        ]



def predict(image):

    interpreter = load_ai()

    labels = load_labels()


    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    size = input_details[0]["shape"][1]


    img = image.resize(
        (size,size)
    )


    img = np.array(img)

    img = np.expand_dims(
        img,
        axis=0
    )


    img = img.astype(
        np.float32
    ) / 255.0



    interpreter.set_tensor(
        input_details[0]["index"],
        img
    )


    interpreter.invoke()


    output = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]


    index = np.argmax(output)

    confidence = float(
        output[index] * 100
    )


    return labels[index], confidence



# ---------------- SESSION ----------------


if "logged" not in st.session_state:
    st.session_state.logged=False


if "user" not in st.session_state:
    st.session_state.user=""



# ---------------- LOGIN ----------------


if not st.session_state.logged:


    st.title(
        "🌱 Smart Plant AI"
    )


    option = st.radio(
        "Choose",
        [
            "Login",
            "Register"
        ]
    )


    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )



    if option=="Register":


        if st.button("Create Account"):

            if register(username,password):

                st.success(
                    "Account created"
                )

            else:

                st.error(
                    "Username already exists"
                )



    else:


        if st.button("Login"):


            if check_login(username,password):

                st.session_state.logged=True
                st.session_state.user=username

                st.rerun()

            else:

                st.error(
                    "Invalid login"
                )



# ---------------- MAIN APP ----------------


else:


    st.sidebar.success(
        st.session_state.user
    )


    page = st.sidebar.selectbox(
        "Menu",
        [
            "Dashboard",
            "Analyze Leaf",
            "History"
        ]
    )



    if page=="Dashboard":

        st.title(
            "🌿 Plant Disease Dashboard"
        )

        st.info(
            """
            Upload a leaf image to get:
            
            • Disease prediction
            • Confidence score
            • Treatment advice
            • Prevention tips
            """
        )



    elif page=="Analyze Leaf":


        st.title(
            "🔍 AI Plant Analysis"
        )


        file = st.file_uploader(
            "Upload leaf image",
            type=[
                "png",
                "jpg",
                "jpeg"
            ]
        )


        if file:


            image = Image.open(file)


            st.image(
                image,
                caption="Leaf Image",
                use_container_width=True
            )



            if st.button(
                "Analyze"
            ):


                with st.spinner(
                    "AI analyzing..."
                ):


                    disease,confidence = predict(image)



                st.success(
                    "Analysis Complete"
                )


                st.subheader(
                    "AI Prediction"
                )


                st.write(
                    "Disease:",
                    disease
                )


                st.write(
                    "Confidence:",
                    f"{confidence:.2f}%"
                )



                st.subheader(
                    "Recommendation"
                )


                st.write(
                    """
                    Remove infected leaves,
                    improve crop hygiene,
                    and apply suitable treatment.
                    """
                )


                users=load_users()


                users[
                    st.session_state.user
                ]["history"].append(
                    {
                        "date":str(datetime.now()),
                        "result":disease,
                        "confidence":confidence
                    }
                )


                save_users(users)



    elif page=="History":


        st.title(
            "📄 Analysis History"
        )


        users=load_users()


        history=users[
            st.session_state.user
        ]["history"]


        for item in history:

            st.write(item)



    if st.sidebar.button(
        "Logout"
    ):

        st.session_state.logged=False
        st.session_state.user=""

        st.rerun()
