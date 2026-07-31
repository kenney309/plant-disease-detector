import streamlit as st
import json
import hashlib
import os

# App settings
st.set_page_config(
    page_title="Smart Plant Disease Detector",
    page_icon="🌱",
    layout="centered"
)

USER_FILE = "users.json"


# Create users file if missing
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)


# Password encryption
def encrypt_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Load users
def get_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)


# Save users
def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)


# Register
def register_user(username, password):

    users = get_users()

    if username in users:
        return False

    users[username] = encrypt_password(password)

    save_users(users)

    return True


# Login
def check_login(username, password):

    users = get_users()

    if username in users:

        if users[username] == encrypt_password(password):
            return True

    return False



# Session
if "login" not in st.session_state:
    st.session_state.login = False

if "user" not in st.session_state:
    st.session_state.user = ""



# Login/Register page
if not st.session_state.login:


    st.title("🌱 Smart Plant Disease Detector")

    option = st.radio(
        "Select option",
        ["Login", "Register"]
    )


    if option == "Register":

        st.subheader("Create Account")

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )


        if st.button("Register"):

            if username and password:

                if register_user(username,password):

                    st.success(
                        "Account created. You can now login."
                    )

                else:

                    st.error(
                        "Username already exists."
                    )

            else:

                st.warning(
                    "Fill all fields."
                )



    else:

        st.subheader("Login")

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )


        if st.button("Login"):

            if check_login(username,password):

                st.session_state.login = True
                st.session_state.user = username

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )



# Main application
else:

    st.title("🌿 AI Plant Disease Detector")

    st.success(
        f"Welcome {st.session_state.user}"
    )


    uploaded = st.file_uploader(
        "Upload plant leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded:

        st.image(
            uploaded,
            caption="Uploaded Leaf",
            use_container_width=True
        )

        st.info(
            "AI prediction system will be connected here."
        )


    if st.button("Logout"):

        st.session_state.login = False
        st.session_state.user = ""

        st.rerun()
