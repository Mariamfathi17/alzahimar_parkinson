import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# ------------------------------
# تحميل الموديل (Keras)
# ------------------------------
model = tf.keras.models.load_model("mlp_model_final.keras")

# ------------------------------
# دالة الـ preprocessing
# ------------------------------
def preprocess_image(image):
    img = np.array(image)

    # تحويل للصورة الملونة → BGR → Gray (لو الموديل عايز Gray)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224

    # Normalization
    img = img / 255.0

    # لو الموديل MLP محتاج Flatten: لازم نعمل reshape
    img = img.reshape(1, -1)  

    return img

# ------------------------------
# واجهة Streamlit
# ------------------------------
st.title("🧠 Alzheimer & Parkinson MRI Classifier (Keras Model)")
st.write("Upload an MRI image to predict the disease")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # معالجة الصورة
    processed_img = preprocess_image(image)

    # توقع النتيجة
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # الماب بتاع الكلاسات
    label_map = {0: "Alzheimer", 1: "Normal", 2: "Parkinson"}

    # عرض النتيجة
    st.subheader("Prediction Result:
