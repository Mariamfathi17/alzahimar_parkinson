import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import urllib.request
import os

# --------- تحميل الموديل من GitHub ---------
MODEL_URL = "https://github.com/USERNAME/REPO/raw/main/mlp_model_final.keras"
MODEL_PATH = "mlp_model_final.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

# --------- label map ---------
label_map = {0: "Alzheimer", 1: "Normal", 2: "Parkinson"}

# --------- دالة الـ preprocessing للصورة ---------
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # تحويل من PIL → OpenCV
    img = cv2.resize(img, (224, 224))  # resize
    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # batch size 1
    return img

# --------- Streamlit UI ---------
st.title("🧠 Brain MRI Classifier")
st.write("Upload an MRI image to classify: Alzheimer, Normal, or Parkinson.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)

    # Preprocess
    img = preprocess_image(st.image(uploaded_file).image)

    # Predict
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("Prediction Result")
    st.write(f"**Class:** {label_map[class_idx]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
