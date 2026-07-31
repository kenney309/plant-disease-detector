import streamlit as st
import json
import hashlib
import os
import time
from datetime import datetime


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Smart Plant AI",
    page_icon="🌱",
    layout="wide"
)


USER_FILE = "users.json"


# ---------------- CREATE DATABASE ----------------

if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as file:
        json.dump({}, file)



def load_users():
    with open(USER_FILE, "r") as file:
        return json.load(file)



def save_users(users):
    with open(USER_FILE, "w") as file:
        json.dump(users, file, indent=4)



def hash_password(password):
    return hashlib.sha256(
        password.encode()
    ).hexdigest()



# ---------------- ACCOUNT FUNCTIONS ----------------


def register_user(username, password):

    users = load_users()

    if username in users:
        return False

    users[username] = {
        "password": hash_password(password),
        "reports": []
    }

    save_users(users)

    return True



def login_user(username,password):

    users = load_users()

    if username in users:

        if users[username]["password"] == hash_password(password):
            return True

    return False



# ---------------- SESSION ----------------


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


if "username" not in st.session_state:
    st.session_state.username = ""



# ---------------- LOGIN SYSTEM ----------------


if not st.session_state.logged_in:


    st.title("🌱 Smart Plant AI")

    st.subheader(
        "AI Powered Plant Disease Detection System"
    )


    option = st.radio(
        "Select option",
        [
            "Login",
            "Register"
        ],
        horizontal=True
    )


    if option == "Register":


        st.header("Create Account")


        username = st.text_input(
            "Username"
        )


        password = st.text_input(
            "Password",
            type="password"
        )


        if st.button("Register"):


            if register_user(username,password):

                st.success(
                    "Account created. Login now."
                )

            else:

                st.error(
                    "Username already exists."
                )



    else:


        st.header("Login")


        username = st.text_input(
            "Username"
        )


        password = st.text_input(
            "Password",
            type="password"
        )


        if st.button("Login"):


            if login_user(username,password):

                st.session_state.logged_in = True
                st.session_state.username = username

                st.rerun()

            else:

                st.error(
                    "Incorrect username or password"
                )



# ---------------- MAIN APP ----------------


else:


    st.sidebar.title(
        "🌿 Smart Plant AI"
    )


    page = st.sidebar.selectbox(
        "Navigation",
        [
            "Dashboard",
            "Detect Disease",
            "My Reports",
            "About"
        ]
    )


    st.sidebar.success(
        "User: " + st.session_state.username
    )



    # Dashboard

    if page == "Dashboard":


        st.title(
            "🌱 Dashboard"
        )


        col1,col2,col3 = st.columns(3)


        with col1:
            st.metric(
                "System",
                "Online"
            )


        with col2:
            st.metric(
                "AI",
                "Ready"
            )


        with col3:
            st.metric(
                "Reports",
                "0"
            )


        st.divider()


        st.info(
            """
            Upload a plant leaf image to receive:
            
            • Disease detection
            • Confidence score
            • Treatment advice
            • Prevention methods
            """
        )



    # Detector


    elif page == "Detect Disease":


        st.title(
            "🔍 Plant Disease Detector"
        )


        image = st.file_uploader(
            "Upload leaf image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )


        if image:


            st.image(
                image,
                caption="Uploaded Leaf",
                use_container_width=True
            )


            if st.button(
                "Analyze Image"
            ):


                progress = st.progress(0)


                for i in range(100):

                    time.sleep(0.01)
                    progress.progress(i+1)



                st.success(
                    "Analysis completed"
                )


                st.warning(
                    "AI model will be connected next."
                )



    # Reports


    elif page == "My Reports":


        st.title(
            "📄 My Reports"
        )


        st.info(
            "Your previous plant analysis reports will appear here."
        )



    # About


    elif page == "About":


        st.title(
            "About Smart Plant AI"
        )


        st.write(
            """
            Smart Plant AI is an intelligent agriculture
            assistant designed to help identify plant diseases.
            
            Future features:
            - Real AI diagnosis
            - Treatment recommendations
            - PDF reports
            - Disease history
            - Analytics
            """
        )



    if st.sidebar.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.username = ""

        st.rerun()
