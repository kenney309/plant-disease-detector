import streamlit as st
import json
import hashlib
import os
from datetime import datetime

# Page settings
st.set_page_config(
    page_title="Smart Plant Disease Detector",
    page_icon="🌱",
    layout="centered"
)

USER_FILE = "users.json"


# ---------------- USER SYSTEM ----------------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as file:
            return json.load(file)
    return {}


def save_users(users):
    with open(USER_FILE, "w") as file:
        json.dump(users, file, indent=4)


def register(username, password):
    users = load_users()

    if username in users:
        return False, "Username already exists"

    users[username] = {
        "password": hash_password(password),
        "joined": str(datetime.now())
    }

    save_users(users)
    return True, "Registration successful"


def login(username, password):
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


# ---------------- LOGIN PAGE ----------------

def login_page():

    st.title("🌱 Smart Plant Disease Detector")

    menu = st.selectbox(
        "Choose option",
        ["Login", "Register"]
    )

    if menu == "Register":

        st.subheader("Create Account")

        username = st.text_input("Username")
        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Register"):

            if username and password:
                success, message = register(
                    username,
                    password
                )

                if success:
                    st.success(message)
                else:
                    st.error(message)

            else:
                st.warning("Fill all fields")


    else:

        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            if login(username, password):

                st.session_state.logged_in = True
                st.session_state.username = username

                st.success("Login successful")
                st.rerun()

            else:
                st.error("Wrong username or password")


# ---------------- MAIN APP ----------------

def detector_page():

    st.title("🌿 AI Plant Disease Detector")

    st.write(
        f"Welcome {st.session_state.username}"
    )

    st.info(
        "Upload a plant leaf image to detect possible diseases."
    )

    uploaded_file = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png"]
    )


    if uploaded_file:

        st.image(
            uploaded_file,
            caption="Uploaded Leaf",
            use_container_width=True
        )

        st.warning(
            "AI model connection will be added next."
        )


    if st.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()



# ---------------- RUN ----------------

if st.session_state.logged_in:
    detector_page()

else:
    login_page()
