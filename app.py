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
MODEL_URL = "https://drive.google.com/uc?id=15Kfi84AOr76Ul3o-jMdUZMXLQvL4LpRR"
MODEL_PATH = "alz_parkinson_model.pth"

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
        st.success("Model downloaded successfully!")

    obj = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(obj, dict):
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 3)  # 3 classes: Normal, Alzheimer's, Parkinson's
        model.load_state_dict(obj, strict=False)
        model = model.to(DEVICE)
        model.eval()
        return model
    elif isinstance(obj, nn.Module):
        obj.to(DEVICE).eval()
        return obj
    else:
        raise TypeError(f"Unsupported model format: {type(obj)}")

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])

st.title("Alzheimer’s vs Parkinson’s MRI Classifier")
st.markdown("Upload an MRI scan and the model will predict if it's **Normal**, **Alzheimer’s**, or **Parkinson’s**.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
label_map = {0: "Normal", 1: "Alzheimer’s", 2: "Parkinson’s"}

if uploaded_file:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)

    probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    st.subheader(f"**Prediction:** {label_map.get(pred_idx, 'Unknown')}")

    st.subheader("Class Probabilities")
    for idx, cls in label_map.items():
        st.write(f"- {cls}: {probs[idx]*100:.2f}%")

    st.bar_chart({label_map[i]: float(probs[i]) for i in label_map if i < len(probs)})
