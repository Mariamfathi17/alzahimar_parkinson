import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pandas as pd

# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="Neuro Prediction Platform", layout="wide")

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_mri_model():
    return load_model("alzheimer_cnn_model.h5")

@st.cache_resource
def load_hd_model():
    model = torch.load("dna_model.pth", map_location="cpu")
    model.eval()
    return model

mri_model = load_mri_model()
hd_model = load_hd_model()

# Labels for MRI model
class_labels = ["Alzheimer", "MCI", "Parkinson"]

# -------------------------------
# Helper: DNA One-hot Encoding
# -------------------------------
def dna_one_hot(seq, max_len=100):
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    one_hot = np.zeros((4, max_len))
    for i, nucleotide in enumerate(seq[:max_len]):
        if nucleotide in mapping:
            one_hot[mapping[nucleotide], i] = 1
    return one_hot

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2 = st.tabs(["🧠 MRI Prediction", "🧬 Huntington’s Disease DNA"])

# -------------------------------
# Tab 1: MRI Prediction
# -------------------------------
with tab1:
    st.title("🧠 Alzheimer & Parkinson MRI Prediction")

    uploaded_file = st.file_uploader("📂 Upload an MRI image", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((128,128))
        x = np.expand_dims(np.array(img_resized)/255.0, axis=0)

        preds = mri_model.predict(x)
        predicted_class = np.argmax(preds[0])
        confidence = preds[0][predicted_class]

        st.image(img_resized, caption="Uploaded MRI", use_column_width=True)
        st.subheader(f"Prediction: **{class_labels[predicted_class]}** ({confidence:.2f})")

        st.bar_chart(pd.DataFrame({"Probability": preds[0]}, index=class_labels))

# -------------------------------
# Tab 2: Huntington’s DNA
# -------------------------------
with tab2:
    st.title("🧬 Genetic Forecasting of Huntington’s Disease")

    uploaded_file = st.file_uploader("📂 Upload a DNA sequence file", type=["csv", "txt"], key="dna")
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            if "sequence" not in df.columns:
                st.error("❌ CSV must contain a 'sequence' column.")
            else:
                st.success(f"✅ Loaded {len(df)} DNA sequences.")
                results = []
                for seq in df["sequence"]:
                    x = dna_one_hot(seq, max_len=100)
                    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        pred = hd_model(x_tensor).item()
                    label = "🧬 At Risk" if pred > 0.5 else "✅ Healthy"
                    results.append({"Sequence": seq[:30]+"...", "Prediction": label, "Score": round(pred, 3)})
                st.dataframe(pd.DataFrame(results))

        elif uploaded_file.name.endswith(".txt"):
            seq = uploaded_file.read().decode("utf-8").strip()
            st.code(seq, language="text")
            x = dna_one_hot(seq, max_len=100)
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = hd_model(x_tensor).item()
            if pred > 0.5:
                st.error("🧬 Prediction: At Risk of Huntington’s Disease")
            else:
                st.success("✅ Prediction: Healthy")

# -------------------------------
# Contact Info
# -------------------------------
st.sidebar.subheader("📞 Contact Info")
st.sidebar.write("**Name:** Nahrwan Thaer")
st.sidebar.write("**Email:** nahrwanthaer@gmail.com")
st.sidebar.write("**Phone:** +90 534 078 62 26")
