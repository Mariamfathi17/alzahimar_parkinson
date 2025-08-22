import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import streamlit as st

# ----------------------------
# 1. تحميل الموديل
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# نفس الـ architecture اللي اتدرب بيه (ResNet18)
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # 3 classes: Normal, Alzheimer's, Parkinson's

# تحميل weights من الملف
model.load_state_dict(torch.load("https://drive.google.com/file/d/15Kfi84AOr76Ul3o-jMdUZMXLQvL4LpRR/view?usp=drive_link", map_location=device))
model.to(device)
model.eval()

# ----------------------------
# 2. Label map
# ----------------------------
label_map = {0: "Normal", 1: "Alzheimer's", 2: "Parkinson's"}

# ----------------------------
# 3. Preprocessing للصور
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# ----------------------------
# 4. دالة prediction
# ----------------------------
def predict(image):
    img = Image.open(image).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    
    return label_map[preds.item()]

# ----------------------------
# 5. Streamlit App
# ----------------------------
st.title("🧠 MRI Classifier (Alzheimer's / Parkinson's / Normal)")

uploaded_file = st.file_uploader("ارفع صورة MRI", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="الصورة المرفوعة", use_column_width=True)

    prediction = predict(uploaded_file)
    st.success(f"✅ Prediction: {prediction}")
