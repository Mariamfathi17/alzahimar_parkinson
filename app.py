import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# تحميل الموديل
model = torch.load("model.pth", map_location=torch.device('cpu'))
model.eval()

st.title("Alzheimer vs Parkinson Prediction (CNN)")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    classes = ["Alzheimer", "Parkinson", "Normal"]
    st.write("Prediction:", classes[predicted.item()])
