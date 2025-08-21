# app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib

@st.cache_resource
def load_models():
    main_model = joblib.load("main_model.pkl")   # Normal / Alzheimer / Parkinson
    alzheimer_model = joblib.load("alz_severity.pkl")  # Severity levels
    parkinson_model = joblib.load("parkinson_stage.pkl")  # Stages
    return main_model, alzheimer_model, parkinson_model

main_model, alzheimer_model, parkinson_model = load_models()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.title("🧠 MRI Triage App")
st.write("Upload an MRI scan to predict **Normal / Alzheimer’s / Parkinson’s**")

uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # -------------------------------
    # 4. Prediction from main model
    # -------------------------------
    main_pred = main_model.predict(img_tensor.view(img_tensor.size(0), -1))
    classes = ["Normal", "Alzheimer", "Parkinson"]
    main_result = classes[main_pred[0]]

    st.subheader(f"🧾 Primary Prediction: {main_result}")

    # -------------------------------
    # 5. If Alzheimer → Severity
    # -------------------------------
    if main_result == "Alzheimer":
        severity_pred = alzheimer_model.predict(img_tensor.view(img_tensor.size(0), -1))
        severity_classes = ["Mild", "Moderate", "Severe"]
        severity_result = severity_classes[severity_pred[0]]
        st.success(f"Alzheimer Severity: {severity_result}")

    # -------------------------------
    # 6. If Parkinson → Stage
    # -------------------------------
    elif main_result == "Parkinson":
        stage_pred = parkinson_model.predict(img_tensor.view(img_tensor.size(0), -1))
        stage_classes = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]
        stage_result = stage_classes[stage_pred[0]]
        st.success(f"Parkinson Stage: {stage_result}")

    else:
        st.info("The scan looks Normal ✅")
