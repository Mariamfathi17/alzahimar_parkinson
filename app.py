import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import torch

# ----------------- Page Setup -----------------
st.set_page_config(page_title="Neuro Prediction Platform", layout="wide")
st.title("🧠🧬 Neuro Prediction Platform")

# ----------------- Sidebar Navigation -----------------
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Choose a Module", ["MRI Prediction", "DNA Prediction"])

# ======================================================
# 🧠 Alzheimer & Parkinson MRI Prediction
# ======================================================
if choice == "MRI Prediction":
    st.header("🧠 Alzheimer & Parkinson MRI Prediction")

    # Load MRI Model
    @st.cache_resource
    def load_mri_model():
        return load_model("alzheimer_cnn_model.h5")

    mri_model = load_mri_model()
    class_labels = ["Alzheimer", "MCI", "Parkinson"]

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

# ======================================================
# 🧬 Huntington’s DNA Prediction
# ======================================================
# ======================================================
# 🧬 Huntington’s DNA Prediction
# ======================================================
elif choice == "DNA Prediction":
    st.header("🧬 Genetic Forecasting of Huntington’s Disease")

    # ✅ Define same architecture used during training
    class DNA_Net(nn.Module):
        def __init__(self, input_size=400, hidden_size=128, output_size=1):
            super(DNA_Net, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            x = x.view(1, -1)  # flatten sequence
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return self.sigmoid(x)

    # ✅ Load model with state_dict
    @st.cache_resource
    def load_hd_model():
        model = DNA_Net()
        state_dict = torch.load("dna_model.pth", map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    dna_model = load_hd_model()

    # Helper: One-hot Encoding
    def dna_one_hot(seq, max_len=100):
        mapping = {'A':0, 'C':1, 'G':2, 'T':3}
        one_hot = np.zeros((4, max_len))
        for i, nucleotide in enumerate(seq[:max_len]):
            if nucleotide in mapping:
                one_hot[mapping[nucleotide], i] = 1
        return one_hot.flatten()  # flatten to match input_size

    uploaded_file = st.file_uploader("📂 Upload a DNA sequence file", type=["csv", "txt"])

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
                    x_tensor = torch.tensor(x, dtype=torch.float32)
                    with torch.no_grad():
                        pred = dna_model(x_tensor).item()
                    label = "🧬 At Risk" if pred > 0.5 else "✅ Healthy"
                    results.append({"Sequence": seq[:30]+"...", "Prediction": label, "Score": round(pred, 3)})
                st.dataframe(pd.DataFrame(results))

        elif uploaded_file.name.endswith(".txt"):
            seq = uploaded_file.read().decode("utf-8").strip()
            st.code(seq, language="text")
            x = dna_one_hot(seq, max_len=100)
            x_tensor = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                pred = dna_model(x_tensor).item()
            if pred > 0.5:
                st.error("🧬 Prediction: At Risk of Huntington’s Disease")
            else:
                st.success("✅ Prediction: Healthy")

# ======================================================
# 📞 Contact Info
# ======================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📞 Contact Info")
st.sidebar.write("**Name:** Nahrwan Thaer")
st.sidebar.write("**Email:** nahrwanthaer@gmail.com")
st.sidebar.write("**Phone:** +90 534 078 62 26")
