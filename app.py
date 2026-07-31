import streamlit as st
import json
import hashlib
import os
from datetime import datetime


# ---------------- APP SETTINGS ----------------

st.set_page_config(
    page_title="Smart Plant Disease Detector",
    page_icon="🌱",
    layout="wide"
)


USER_FILE = "users.json"


# ---------------- DATABASE ----------------

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


# ---------------- AUTHENTICATION ----------------

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



def login(username, password):

    users = load_users()

    if username in users:

        if users[username]["password"] == hash_password(password):
            return True

    return False



# ---------------- SESSION ----------------

if "logged" not in st.session_state:
    st.session_state.logged = False

if "username" not in st.session_state:
    st.session_state.username = ""



# ---------------- LOGIN / REGISTER ----------------


if not st.session_state.logged:


    st.title("🌱 Smart Plant Disease Detector")

    choice = st.selectbox(
        "Select action",
        [
            "Login",
            "Register"
        ]
    )


    if choice == "Register":

        st.header("Create Account")


        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )


        if st.button("Create Account"):

            if register(username,password):

                st.success(
                    "Account created successfully. Login now."
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

            if login(username,password):

                st.session_state.logged = True
                st.session_state.username = username

                st.rerun()

            else:

                st.error(
                    "Incorrect login details"
                )



# ---------------- MAIN APP ----------------


else:


    st.sidebar.title(
        "Navigation"
    )


    page = st.sidebar.radio(
        "Go to",
        [
            "Dashboard",
            "Disease Detector",
            "History",
            "Settings"
        ]
    )


    st.sidebar.success(
        f"User: {st.session_state.username}"
    )



    if page == "Dashboard":

        st.title("🌿 Dashboard")

        st.write(
            "Welcome to your Smart Plant Disease Detector."
        )


        st.info(
            "Upload plant leaves and get AI-powered disease analysis."
        )



    elif page == "Disease Detector":


        st.title(
            "🌱 Plant Disease Detection"
        )


        image = st.file_uploader(
            "Upload plant leaf image",
            type=[
                "png",
                "jpg",
                "jpeg"
            ]
        )


        if image:


            st.image(
                image,
                caption="Uploaded Leaf",
                use_container_width=True
            )


            st.warning(
                "AI model will be connected here next."
            )


            prediction = {
                "date": str(datetime.now()),
                "result": "Pending AI Model"
            }


            users = load_users()


            users[
                st.session_state.username
            ]["history"].append(prediction)


            save_users(users)




    elif page == "History":


        st.title(
            "📄 Prediction History"
        )


        users = load_users()


        history = users[
            st.session_state.username
        ]["history"]


        if history:

            for item in history:

                st.write(
                    item
                )


        else:

            st.info(
                "No predictions yet."
            )



    elif page == "Settings":


        st.title(
            "⚙ Settings"
        )


        st.write(
            "Future features:"
        )

        st.write(
            """
            - Google Login
            - Face ID / Fingerprint
            - Notifications
            - Offline Mode
            - Advanced Analytics
            """
        )



    if st.sidebar.button("Logout"):

        st.session_state.logged = False
        st.session_state.username = ""

        st.rerun()
