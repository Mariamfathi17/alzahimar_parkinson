import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# -----------------------------
# إعدادات
# -----------------------------
img_size = 224

def crop_images(img, threshold=10):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    mask = gray > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

def preprocess_images(img, img_size=(img_size, img_size)):
    img = crop_images(img)
    img = cv2.resize(img, img_size)
    # تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=25.0, tileGridSize=(4,4))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img

# -----------------------------
# تحميل الموديل
# -----------------------------
model = load_model("cnn_model.keras")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Alzheimer's / Parkinson's / Normal MRI Classifier")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # قراءة الصورة
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # صورة ملونة
    
    st.image(img, caption="الصورة المرفوعة", use_container_width=True)

    # تطبيق الـ preprocessing
    processed_img = preprocess_images(img, (img_size, img_size))

    # تحويل الصورة لتنسيق مناسب للموديل
    img_array = processed_img.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    # التنبؤ
    prediction = model.predict(img_array)

    st.subheader("نتيجة التنبؤ")
    st.write(prediction)
