import os
import requests
import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("Sentiment Analysis (BERT)")

text = st.text_area("Enter text")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter text")
    else:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text}
        )

        if response.status_code == 200:
            data = response.json()
            st.success(f"Sentiment: {data['sentiment']}")
            st.info(f"Confidence: {data['confidence']:.2f}")
        else:
            st.error(response.text)
