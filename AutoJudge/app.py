import streamlit as st
import joblib
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

# -------------------------
# Load Models
# -------------------------
classifier = joblib.load("models/classifier.pkl")
regressor = joblib.load("models/regressor.pkl")
tfidf = joblib.load("models/tfidf.pkl")
scaler = joblib.load("models/scaler.pkl")

# -------------------------
# Text Preprocessing
# -------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("🤖 AutoJudge")
st.subheader("Predict Programming Problem Difficulty")

st.markdown("Enter the problem details below:")

title = st.text_input("Problem Title")
description = st.text_area("Problem Description", height=200)
input_desc = st.text_area("Input Description", height=150)
output_desc = st.text_area("Output Description", height=150)

if st.button("Predict Difficulty"):
    if not description.strip():
        st.warning("Please enter at least the problem description.")
    else:
        combined_text = f"{title} {description} {input_desc} {output_desc}"
        combined_text = preprocess_text(combined_text)

        text_tfidf = tfidf.transform([combined_text])
        text_len = scaler.transform([[len(combined_text)]])
        X = hstack([text_tfidf, csr_matrix(text_len)])

        class_pred = classifier.predict(X)[0]
        score_pred = regressor.predict(X)[0]

        class_map = {0: "Hard", 1: "Medium", 2: "Easy"}

        st.success("Prediction Complete ✅")
        st.markdown(f"### 🧠 Predicted Difficulty: **{class_map[class_pred]}**")
        st.markdown(f"### ⭐ Predicted Score: **{score_pred:.2f} / 10**")
