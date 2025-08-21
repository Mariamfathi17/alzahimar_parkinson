import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os

# رابط Google Drive للموديل
model_url = "https://drive.google.com/uc?id=15Kfi84AOr76Ul3o-jMdUZMXLQvL4LpRR"
model_path = "alz_parkinson_model.pth"

# تحميل الموديل من Google Drive لو مش موجود
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# تحميل الموديل
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()

# لو الموديل متسجل بـ state_dict فقط
if isinstance(model, dict):
    st.error("الموديل متسجل بـ state_dict فقط. لازم يكون متسجل بكامل الكلاس.")
else:
    st.success("✅ الموديل اتحمّل بنجاح")

# دالة لتحويل الصورة
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # لو الموديل اتدرّب كده
])

# Streamlit UI
st.title("🧠 MRI Classification: Alzheimer's vs Parkinson's")

uploaded_file = st.file_uploader("ارفع صورة MRI هنا", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة اللي رفعتها", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    classes = ["Alzheimer's", "Parkinson's"]
    prediction = classes[predicted.item()]

    st.subheader(f"🔍 التنبؤ: {prediction}")
