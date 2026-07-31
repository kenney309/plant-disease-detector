st.markdown("""
<style>

.stApp {
    background:#f7faf7;
}


.header {
    background:#166534;
    padding:35px;
    border-radius:0 0 25px 25px;
    text-align:center;
    color:white;
}


.header h1 {
    font-size:42px;
    margin-bottom:5px;
}


.header p {
    font-size:18px;
}



.login-box {

    background:white;
    padding:35px;
    border-radius:18px;
    box-shadow:0 8px 25px rgba(0,0,0,0.08);

}



.info-card {

    background:#ecfdf5;
    padding:18px;
    border-radius:15px;
    margin-top:15px;

}



.stButton button {

    background:#166534;
    color:white;
    border-radius:10px;
    height:45px;

}


.stButton button:hover {

    background:#15803d;

}


</style>
""",
unsafe_allow_html=True)



st.markdown(
"""
<div class="header">

<h1>🌱 Smart Plant AI</h1>

<p>
Your intelligent assistant for plant health monitoring
</p>

</div>
""",
unsafe_allow_html=True
)



st.write("")



left,right = st.columns(
    [1,1]
)



with left:

    st.markdown(
    """
    <div class="info-card">

    🌿 AI Plant Diagnosis  
    <br><br>
    📷 Leaf Image Analysis  
    <br><br>
    📄 Digital Reports  
    <br><br>
    🌾 Smart Farming Advice

    </div>
    """,
    unsafe_allow_html=True
    )



with right:

    st.markdown(
    "<div class='login-box'>",
    unsafe_allow_html=True
    )


    option = st.radio(
        "",
        [
            "Login",
            "Register"
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


    if option=="Register":

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
