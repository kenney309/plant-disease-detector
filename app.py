import streamlit as st
import json
import hashlib
import os
from datetime import datetime


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Smart Plant AI",
    page_icon="🌱",
    layout="wide"
)


# ---------------- DESIGN ----------------

st.markdown("""
<style>

.stApp{
    background:#f4faf4;
}


.header{
    background:#166534;
    padding:35px;
    border-radius:0 0 25px 25px;
    color:white;
    text-align:center;
}


.header h1{
    font-size:45px;
}


.header p{
    font-size:18px;
}


.card{

    background:white;
    padding:30px;
    border-radius:20px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.1);

}


.stButton button{

    background:#166534;
    color:white;
    border-radius:12px;
    height:45px;
    width:100%;

}


.stButton button:hover{

    background:#15803d;

}


</style>
""", unsafe_allow_html=True)



# ---------------- DATABASE ----------------


USER_FILE="users.json"


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



# ---------------- ACCOUNT SYSTEM ----------------


def create_account(username,password):

    users=load_users()

    if username in users:
        return False


    users[username]={
        "password":hash_password(password),
        "history":[]
    }


    save_users(users)

    return True



def check_login(username,password):

    users=load_users()


    if username in users:

        return users[username]["password"] == hash_password(password)


    return False



# ---------------- SESSION ----------------


if "logged" not in st.session_state:

    st.session_state.logged=False



if "username" not in st.session_state:

    st.session_state.username=""




# =================================================
# LOGIN PAGE
# =================================================


if not st.session_state.logged:


    st.markdown("""
    <div class="header">

    <h1>🌱 Smart Plant AI</h1>

    <p>
    Intelligent Plant Disease Detection System
    </p>

    </div>
    """,
    unsafe_allow_html=True
    )


    st.write("")


    left,right=st.columns([1,1])


    with left:

        st.markdown("""
        <div class="card">

        🌿 AI Plant Diagnosis

        <br><br>

        📷 Leaf Image Analysis

        <br><br>

        📊 Smart Reports

        <br><br>

        🌾 Farming Recommendations

        </div>
        """,
        unsafe_allow_html=True
        )


    with right:


        st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True
        )


        option=st.radio(
            "Account",
            [
                "Login",
                "Register"
            ],
            horizontal=True
        )


        username=st.text_input(
            "Username"
        )


        password=st.text_input(
            "Password",
            type="password"
        )



        if option=="Register":


            if st.button("Create Account"):


                if create_account(username,password):

                    st.success(
                        "Account created. Login now."
                    )

                else:

                    st.error(
                        "Username already exists"
                    )



        else:


            if st.button("Login"):


                if check_login(username,password):

                    st.session_state.logged=True

                    st.session_state.username=username

                    st.rerun()


                else:

                    st.error(
                        "Incorrect username or password"
                    )


        st.markdown(
        "</div>",
        unsafe_allow_html=True
        )




# =================================================
# MAIN APPLICATION
# =================================================


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
            "History"
        ]
    )


    st.sidebar.success(
        "User: "+st.session_state.username
    )



    # Dashboard

    if page=="Dashboard":


        st.title(
            "🌱 Dashboard"
        )


        a,b,c=st.columns(3)


        a.metric(
            "AI Status",
            "Ready"
        )


        b.metric(
            "Reports",
            "0"
        )


        c.metric(
            "System",
            "Online"
        )


        st.info("""
        Welcome to Smart Plant AI.

        Upload a plant leaf image and receive:
        
        ✓ Disease detection
        ✓ Confidence score
        ✓ Treatment advice
        ✓ Prevention methods
        """)




    # Profile


    elif page=="Profile":


        st.title(
            "👤 Profile"
        )


        st.write(
            "Username:",
            st.session_state.username
        )


        st.success(
            "Account protected"
        )



    # Plant analysis


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
                caption="Uploaded Leaf",
                use_container_width=True
            )


            if st.button(
                "Analyze Plant"
            ):


                st.warning(
                    "AI model connection will be added here."
                )


                users=load_users()


                users[
                    st.session_state.username
                ]["history"].append(
                    {
                        "date":str(datetime.now()),
                        "result":"Pending AI model"
                    }
                )


                save_users(users)



    # History


    elif page=="History":


        st.title(
            "📄 Analysis History"
        )


        users=load_users()


        history=users[
            st.session_state.username
        ]["history"]


        if history:

            for item in history:

                st.write(item)

        else:

            st.info(
                "No analysis history yet."
            )



    if st.sidebar.button("Logout"):

        st.session_state.logged=False

        st.session_state.username=""

        st.rerun()
