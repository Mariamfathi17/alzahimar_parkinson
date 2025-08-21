import streamlit as st
import pickle
import numpy as np
from PIL import Image
import gdown
import os

# ===============================
# 1. Download model
# ===============================
file_id = "15Kfi84AOr76Ul3o-jMdUZMXLQvL4LpRR"
url = f"https://drive.google.com/uc?id={file_id}"
output = "alz_parkinson_model.pkl"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# ===============================
# 2. Load model
# ===============================
with open(output, "rb") as f:
    model = pickle.load(f)

# ===============================
# 3. Streamlit App
# ===============================
st.title("🧠 Alzheimer vs Parkinson Classifier")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Resize like training
    image = image.resize((224, 224))

    # Convert to numpy
    img_array = np.array(image)

    # Flatten (لو ده اللي اتعمل في التدريب)
    img_flat = img_array.flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(img_flat)[0]

    if prediction == 0:
        st.success("✅ Prediction: Alzheimer's")
    else:
        st.success("✅ Prediction: Parkinson's")
