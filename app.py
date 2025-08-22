import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import urllib.request
import os

# --------- تحميل الموديل من GitHub ---------
MODEL_URL = "https://github.com/Mariamfathi17/alzahimar_parkinson/blob/main/mlp_model_final.keras"
MODEL_PATH = "mlp_model_final.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

# --------- label map ---------
label_map = {0: "Alzheimer", 1: "Normal", 2: "Parkinson"}

# --------- دالة الـ preprocessing للصورة ---------
from PIL import Image
import numpy as np
import cv2
import streamlit as st

def preprocess_image(image):
    # نحول PIL → numpy → OpenCV
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))  # مقاس يناسب الموديل بتاعك
    img = img / 255.0  # Normalization
    return np.expand_dims(img, axis=0)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # نقرأ الصورة بـ PIL
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption="Uploaded Image", use_container_width=True)

    # معالجة الصورة
    img = preprocess_image(pil_image)

    st.write("Image shape after preprocessing:", img.shape)


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
