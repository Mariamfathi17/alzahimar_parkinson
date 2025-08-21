
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import gdown

# ----------------------------------
# 1. Download the model from Google Drive if not yet present
# ----------------------------------
model_url = "https://drive.google.com/uc?id=15Kfi84AOr76Ul3o-jMdUZMXLQvL4LpRR"
model_path = "alzheimers_parkinson_model.pth"

if not os.path.exists(model_path):
    with st.spinner("Downloading model, please wait..."):
        gdown.download(model_url, model_path, quiet=False)
    st.success("Model downloaded successfully!")

# ----------------------------------
# 2. Load the model
# ----------------------------------
@st.cache_resource
def load_model(path):
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model(model_path)

# ----------------------------------
# 3. Define image preprocessing
# ----------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----------------------------------
# 4. Streamlit App UI
# ----------------------------------
st.title("Alzheimer’s vs Parkinson’s MRI Classifier")
st.markdown("Upload an MRI scan to predict if it’s **Normal**, **Alzheimer’s**, or **Parkinson’s**.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)  # add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    # Map numerical prediction to label
    label_map = {
        0: "Normal",
        1: "Alzheimer’s",
        2: "Parkinson’s"
    }
    pred_label = label_map.get(predicted.item(), "Unknown")

    st.subheader("Prediction Result")
    st.success(f"→ **{pred_label}**")

    # Optional: Display probabilities
    probs = torch.softmax(outputs, dim=1).numpy()[0]
    st.subheader("Class Probabilities")
    for idx, cls in label_map.items():
        st.write(f"- {cls}: {probs[idx]*100:.2f}%")

    # Optional: Simple bar chart for probabilities
    st.bar_chart({label_map[idx]: probs[idx] for idx in range(len(probs))})
