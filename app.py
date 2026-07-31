import streamlit as st
from PIL import Image
import requests
import json
import hashlib
import os
from datetime import datetime


st.set_page_config(
    page_title="Smart Plant AI",
    page_icon="🌱",
    layout="wide"
)


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



def register(username,password):

    users = load_users()

    if username in users:
        return False

    users[username] = {
        "password": hash_password(password),
        "history":[]
    }

    save_users(users)

    return True



def login(username,password):

    users = load_users()

    if username in users:
        return users[username]["password"] == hash_password(password)

    return False



# ---------------- SESSION ----------------

if "logged" not in st.session_state:
    st.session_state.logged=False



# ---------------- LOGIN ----------------

if not st.session_state.logged:


    st.title("🌱 Smart Plant AI")


    choice = st.radio(
        "Choose",
        ["Login","Register"],
        horizontal=True
    )


    username = st.text_input("Username")
    password = st.text_input(
        "Password",
        type="password"
    )


    if choice=="Register":

        if st.button("Create Account"):

            if register(username,password):
                st.success("Account created")
            else:
                st.error("Username already exists")



    else:

        if st.button("Login"):

            if login(username,password):

                st.session_state.logged=True
                st.session_state.username=username
                st.rerun()

            else:
                st.error("Wrong username or password")



# ---------------- APP ----------------

else:


    st.sidebar.success(
        st.session_state.username
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

        st.title("🌿 Smart Plant Dashboard")

        st.info(
            """
            Upload a leaf image.
            The AI will analyze possible plant diseases.
            """
        )



    elif page=="Analyze Leaf":


        st.title("🔍 Plant Analysis")


        image = st.file_uploader(
            "Upload leaf image",
            type=["jpg","png","jpeg"]
        )


        if image:


            img = Image.open(image)

            st.image(
                img,
                caption="Uploaded Leaf",
                use_container_width=True
            )


            if st.button("Analyze"):


                st.warning(
                    "AI connection will be added here."
                )


                users = load_users()

                users[
                    st.session_state.username
                ]["history"].append(
                    {
                        "date":str(datetime.now()),
                        "result":"Pending AI model"
                    }
                )

                save_users(users)



    elif page=="History":

        st.title("📄 My Reports")

        users = load_users()

        for item in users[
            st.session_state.username
        ]["history"]:

            st.write(item)



    if st.sidebar.button("Logout"):

        st.session_state.logged=False
        st.rerun()
