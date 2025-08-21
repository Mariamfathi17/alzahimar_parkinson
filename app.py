import os
import io
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gdown

st.set_page_config(page_title="Alzheimer’s vs Parkinson’s MRI Classifier", layout="centered")
DEVICE = torch.device("cpu")

# -----------------------------
# 1) Download the model if not present
# -----------------------------
MODEL_URL = "https://drive.google.com/uc?id=15Kfi84AOr76Ul3o-jMdUZMXLQvL4LpRR"
MODEL_PATH = "alzheimers_parkinson_model.pth"

@st.cache_resource(show_spinner="Loading model from Drive...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
        st.success("Model downloaded successfully!")

    obj = torch.load(MODEL_PATH, map_location=DEVICE)

    # لو الموديل متخزن كـ state_dict
    if isinstance(obj, dict):
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 3)   # عندك 3 classes (Normal, Alzheimer, Parkinson)
        model.load_state_dict(obj, strict=False)
        model.to(DEVICE).eval()
        return model

    # لو متخزن كنموذج كامل
    elif isinstance(obj, nn.Module):
        obj.to(DEVICE).eval()
        return obj

    else:
        raise TypeError(f"Unexpected model format: {type(obj)}")

model = load_model()

# -----------------------------
# 2) Define preprocess transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# 3) UI
# -----------------------------
st.title("Alzheimer’s vs Parkinson’s MRI Classifier")
st.markdown("Upload an MRI scan and the model will predict if it's **Normal**, **Alzheimer’s**, or **Parkinson’s**.")

uploaded = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
label_map = {0: "Normal", 1: "Alzheimer’s", 2: "Parkinson’s"}

if uploaded:
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(img, caption="Uploaded MRI Scan", use_container_width=True)

    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(x)

    probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())

    st.subheader(f"Prediction: **{label_map.get(pred, 'Unknown')}**")

    st.subheader("Class Probabilities")
    for idx, name in label_map.items():
        if idx < len(probs):
            st.write(f"- {name}: {probs[idx]*100:.2f}%")

    st.bar_chart({label_map[i]: float(probs[i]) for i in label_map if i < len(probs)})
