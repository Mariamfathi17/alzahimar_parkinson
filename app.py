import os
import streamlit as st
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import gdown

st.title("MRI Classifier – Alzheimer’s vs Parkinson’s vs Normal")

# تحميل الموديل من Drive
MODEL_URL = "https://drive.google.com/uc?id=15Kfi84AOr76Ul3o-jMdUZMXLQvL4LpRR"
MODEL_PATH = "alzheimers_resnet18.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
    st.success("Model downloaded!")

# تحميل الموديل
device = torch.device("cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

label_map = {0: "Normal", 1: "Alzheimer’s", 2: "Parkinson’s"}

def crop_images(img, threshold=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    mask = gray > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

def preprocess_images(img, img_size=224):
    img = crop_images(img)
    img = cv2.resize(img, (img_size, img_size))
    clahe = cv2.createCLAHE(clipLimit=25.0, tileGridSize=(4,4))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img

uploaded = st.file_uploader("Upload MRI Image", type=["jpg","jpeg","png"])
if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img[:,:,::-1], caption="Original Image", use_container_width=True)

    processed = preprocess_images(img)
    st.image(processed[:,:,::-1], caption="Processed Image", use_container_width=True)

    # Prepare tensor
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    inp = to_tensor(Image.fromarray(processed)).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)
        pred = torch.argmax(out,1).item()
    st.success(f"Prediction: ** {label_map[pred]} **")
