import streamlit as st
import json
import hashlib
import os
from datetime import datetime


st.set_page_config(
    page_title="Smart Plant AI",
    page_icon="🌱",
    layout="wide"
)


# ---------- DESIGN ----------

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #e8f5e9,
        #ffffff
    );
}

.main-title{
    color:#1b5e20;
    text-align:center;
    font-size:45px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#555;
    font-size:18px;
}

.card{
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0 4px 15px rgba(0,0,0,0.1);
}

</style>
""",
unsafe_allow_html=True)



# ---------- DATABASE ----------


FILE="users.json"


if not os.path.exists(FILE):

    with open(FILE,"w") as f:
        json.dump({},f)



def load_users():

    with open(FILE,"r") as f:
        return json.load(f)



def save_users(data):

    with open(FILE,"w") as f:
        json.dump(data,f,indent=4)



def encrypt(password):

    return hashlib.sha256(
        password.encode()
    ).hexdigest()



def register(username,password):

    users=load_users()

    if username in users:
        return False

    users[username]={
        "password":encrypt(password),
        "history":[]
    }

    save_users(users)

    return True



def login(username,password):

    users=load_users()

    if username in users:

        return users[username]["password"]==encrypt(password)

    return False



# ---------- SESSION ----------


if "logged" not in st.session_state:
    st.session_state.logged=False


if "username" not in st.session_state:
    st.session_state.username=""



# ---------- LOGIN PAGE ----------


if not st.session_state.logged:


    st.markdown(
        "<div class='main-title'>🌱 Smart Plant AI</div>",
        unsafe_allow_html=True
    )


    st.markdown(
        "<div class='subtitle'>AI powered plant disease detection system</div>",
        unsafe_allow_html=True
    )


    st.write("")


    option=st.radio(
        "",
        [
            "Login",
            "Register"
        ],
        horizontal=True
    )


    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True
    )


    username=st.text_input(
        "Username"
    )


    password=st.text_input(
        "Password",
        type="password"
    )


    if option=="Register":


        if st.button(
            "Create Account",
            use_container_width=True
        ):


            if register(username,password):

                st.success(
                    "Account created. Login now."
                )

            else:

                st.error(
                    "Username already exists"
                )



    else:


        if st.button(
            "Login",
            use_container_width=True
        ):


            if login(username,password):

                st.session_state.logged=True
                st.session_state.username=username

                st.rerun()

            else:

                st.error(
                    "Wrong username or password"
                )



    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )



# ---------- MAIN APP ----------


else:


    st.sidebar.title(
        "🌿 Smart Plant AI"
    )


    page=st.sidebar.selectbox(
        "Navigation",
        [
            "Dashboard",
            "Profile",
            "Analyze Plant",
            "Reports"
        ]
    )


    st.sidebar.success(
        st.session_state.username
    )


# ---------- DASHBOARD ----------


    if page=="Dashboard":


        st.markdown(
            "<div class='main-title'>Dashboard</div>",
            unsafe_allow_html=True
        )


        c1,c2,c3=st.columns(3)


        c1.metric(
            "AI Status",
            "Online"
        )


        c2.metric(
            "Reports",
            "0"
        )


        c3.metric(
            "Accuracy",
            "AI Ready"
        )


        st.info(
            """
            Welcome to Smart Plant AI.

            Features:
            ✓ Disease detection
            ✓ Plant health analysis
            ✓ Treatment recommendations
            ✓ Digital reports
            """
        )



# ---------- PROFILE ----------


    elif page=="Profile":


        st.title(
            "👤 User Profile"
        )


        st.write(
            "Username:",
            st.session_state.username
        )


        st.success(
            "Account secured"
        )



# ---------- ANALYSIS ----------


    elif page=="Analyze Plant":


        st.title(
            "🔍 Plant Analysis"
        )


        image=st.file_uploader(
            "Upload leaf image",
            type=[
                "png",
                "jpg",
                "jpeg"
            ]
        )


        if image:


            st.image(
                image,
                caption="Uploaded Plant",
                use_container_width=True
            )


            if st.button(
                "Start Analysis"
            ):


                st.warning(
                    "AI model will analyze the leaf here."
                )


                users=load_users()


                users[
                    st.session_state.username
                ]["history"].append(
                    {
                        "date":str(datetime.now()),
                        "result":"Waiting for AI model"
                    }
                )


                save_users(users)



# ---------- REPORTS ----------


    elif page=="Reports":


        st.title(
            "📄 Analysis Reports"
        )


        users=load_users()


        reports=users[
            st.session_state.username
        ]["history"]


        if reports:

            for r in reports:
                st.write(r)

        else:

            st.info(
                "No reports available yet."
            )



    if st.sidebar.button(
        "Logout"
    ):

        st.session_state.logged=False
        st.session_state.username=""

        st.rerun()
