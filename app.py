import streamlit as st
from datetime import datetime

# Page setup
st.set_page_config(
    page_title="Smart Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

# ---------- CSS DESIGN ----------
st.markdown("""
<style>

/* Background */
.stApp {
    background: #f4f8f4;
}

/* Header */
.header {
    background: linear-gradient(135deg,#2d6a4f,#40916c);
    padding: 35px;
    border-radius: 0px 0px 25px 25px;
    text-align:center;
    color:white;
}

.header h1 {
    color:white !important;
    font-size:38px;
}

/* Cards */
.card {
    background:white;
    padding:30px;
    border-radius:25px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.12);
}


/* Text */
h1,h2,h3,h4,p,label {
    color:#1b4332 !important;
}


/* Features */
.feature {
    font-size:18px;
    color:#333333;
    padding:15px;
}


/* Inputs */
input {
    background:white !important;
    color:black !important;
    border:2px solid #40916c !important;
    border-radius:12px !important;
    padding:12px !important;
}


/* Buttons */
.stButton button {

    background:#2d6a4f !important;
    color:white !important;
    border-radius:12px !important;
    padding:10px 25px !important;
    border:none !important;

}


.stButton button:hover {

    background:#1b4332 !important;

}


/* Radio */
.stRadio label {
    color:#1b4332 !important;
}

</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown("""
<div class="header">

<h1>🌿 Smart Plant Disease Detector</h1>

<p style="color:white !important;">
AI Powered Crop Disease Identification System
</p>

</div>
""", unsafe_allow_html=True)



# ---------- MAIN AREA ----------

left,right = st.columns([1,1])


# LEFT SIDE
with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🌱 AI Plant Diagnosis")

    st.markdown("""
    <div class="feature">
    🌿 Detect plant diseases using Artificial Intelligence
    </div>

    <div class="feature">
    📷 Upload leaf images for analysis
    </div>

    <div class="feature">
    📊 Generate smart disease reports
    </div>

    <div class="feature">
    💡 Get treatment recommendations
    </div>
    """, unsafe_allow_html=True)


    st.markdown("</div>",unsafe_allow_html=True)



# RIGHT SIDE LOGIN
with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Account")

    option = st.radio(
        "",
        ["Login","Register"],
        horizontal=True
    )


    if option=="Login":

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )


        if st.button("Login"):

            if username and password:

                st.success("Login successful")

            else:

                st.warning("Enter username and password")


    else:

        username = st.text_input(
            "Create Username"
        )

        password = st.text_input(
            "Create Password",
            type="password"
        )


        if st.button("Register"):

            if username and password:

                st.success("Account created successfully")

            else:

                st.warning("Fill all fields")



    st.markdown("</div>",unsafe_allow_html=True)



# FOOTER

st.markdown("""
<br>
<center>
<p style="color:#555;">
Smart Plant Disease Detector © 2026
</p>
</center>
""",unsafe_allow_html=True)
