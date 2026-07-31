import streamlit as st

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        120deg,
        #d8f3dc,
        #ffffff,
        #b7e4c7
    );
}


/* Main title */

.big-title {
    text-align:center;
    font-size:55px;
    font-weight:800;
    color:#14532d;
    margin-top:30px;
}


.tagline {
    text-align:center;
    font-size:20px;
    color:#374151;
    margin-bottom:30px;
}



/* Login card */

.login-card {

    background:white;
    padding:40px;
    border-radius:25px;
    box-shadow:
    0 10px 30px rgba(0,0,0,0.15);

}



/* Feature cards */

.feature {

    background:#ffffff;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:
    0 5px 15px rgba(0,0,0,0.08);

    font-size:17px;

}



div.stButton > button {

    background:#2d6a4f;
    color:white;
    border-radius:12px;
    height:45px;
    font-size:17px;
    border:none;

}


div.stButton > button:hover {

    background:#40916c;
    color:white;

}


</style>
""",
unsafe_allow_html=True)



st.markdown(
"""
<div class="big-title">
🌱 Smart Plant AI
</div>

<div class="tagline">
Intelligent Plant Disease Detection & Smart Farming Assistant
</div>
""",
unsafe_allow_html=True
)



# Feature row

c1,c2,c3 = st.columns(3)


with c1:
    st.markdown(
    """
    <div class="feature">
    🌿<br>
    AI Disease Detection
    </div>
    """,
    unsafe_allow_html=True
    )


with c2:
    st.markdown(
    """
    <div class="feature">
    📊<br>
    Health Reports
    </div>
    """,
    unsafe_allow_html=True
    )


with c3:
    st.markdown(
    """
    <div class="feature">
    🌾<br>
    Farming Advice
    </div>
    """,
    unsafe_allow_html=True
    )



st.write("")



# Login/Register card

st.markdown(
"<div class='login-card'>",
unsafe_allow_html=True
)


choice = st.radio(
"",
[
"🔐 Login",
"📝 Register"
],
horizontal=True
)


username = st.text_input(
"Username"
)


password = st.text_input(
"Password",
type="password"
)


if choice=="📝 Register":

    st.button(
        "Create Account",
        use_container_width=True
    )


else:

    st.button(
        "Login",
        use_container_width=True
    )


st.markdown(
"</div>",
unsafe_allow_html=True
)
