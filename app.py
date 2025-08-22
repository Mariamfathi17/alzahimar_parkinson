import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image

# ------------------------------
# تحميل الموديل
# ------------------------------
model = pickle.load(open("mlp_model_final.keras", "rb"))

# ------------------------------
# دالة الـ preprocessing
# ------------------------------
def preprocess_image(image):
    # تحويل من PIL → numpy
    img = np.array(image)

    # تحويل للصورة الرمادية
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # تغيير الحجم (نخليها 224x224 مثلاً)
    img = cv2.resize(img, (224, 224))

    # تحويلها إلى شكل مناسب للموديل
    img = img / 255.0   # Normalization
    img = img.reshape(1, 224, 224, 1)  # (Batch, H, W, Channels)

    return img

# ------------------------------
# واجهة Streamlit
# ------------------------------
st.title("🧠 Alzheimer & Parkinson MRI Classifier")
st.write("Upload an MRI image to predict the disease")

# رفع الصورة
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # عرض الصورة
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # معالجة الصورة
    processed_img = preprocess_image(image)

    # توقع النتيجة
    prediction = model.predict(processed_img)

    # عرض النتيجة
    st.subheader("Prediction Result:")
    if prediction[0] == 0:
        st.success("✅ Normal")
    elif prediction[0] == 1:
        st.warning("🧩 Alzheimer Detected")
    else:
        st.error("⚠️ Parkinson Detected")
