import streamlit as st
import pickle
import numpy as np
from PIL import Image
import cv2
import gdown
import os

# ===============================
# 1. Download the model file from Google Drive
# ===============================
file_id = "15Kfi84AOr76Ul3o-jMdUZMXLQvL4LpRR"
url = f"https://drive.google.com/uc?id={file_id}"
output = "alz_parkinson_model.pkl"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# ===============================
# 2. Load the model
# ===============================
with open(output, "rb") as f:
    model = pickle.load(f)

# ===============================
# 3. Streamlit App UI
# ===============================
st.title("🧠 Alzheimer vs Parkinson Classifier")

st.write("Upload an MRI image and the model will predict whether it indicates **Alzheimer's** or **Parkinson's**.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # ===============================
    # 4. Preprocess image
    # ===============================
    img = np.array(image)

    # Resize to 224x224 (or size used during training)
    img_resized = cv2.resize(img, (224, 224))

    # Convert to grayscale if needed
    if len(img_resized.shape) == 3:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Flatten image (depends on how model trained)
    img_flat = img_resized.flatten().reshape(1, -1)

    # ===============================
    # 5. Predict
    # ===============================
    prediction = model.predict(img_flat)[0]

    if prediction == 0:
        st.success("✅ Prediction: Alzheimer's")
    else:
        st.success("✅ Prediction: Parkinson's")
