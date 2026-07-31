import streamlit as st
import json
import hashlib
import os

st.set_page_config(
    page_title="Smart Plant AI",
    page_icon="🌱"
)

st.markdown("""
<style>
.stApp {
    background:#f7faf7;
}
</style>
""", unsafe_allow_html=True)
