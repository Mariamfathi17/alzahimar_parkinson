import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

# تحميل الموديل keras
model = tf.keras.models.load_model("cnn_model.keras")

# حجم الصور اللي اتدرب عليها الموديل
IMG_SIZE = 224  

# دالة قص الصورة (إزالة الخلفية السوداء)
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

# دالة تجهيز الصورة للبريدكشن
def preprocess_images(img, img_size=(IMG_SIZE, IMG_SIZE)):
    img = crop_images(img)
    img = cv2.resize(img, img_size)
    # تحسين الكنتراست باستخدام CLAHE
    clahe = cv2.createCLAHE(clipLimit=25.0, tileGridSize=(4,4))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img

# عنوان الموقع
st.title("🧠 Alzheimer's / Parkinson's / Normal MRI Classifier")

# رفع الصورة من المستخدم
uploaded_file = st.file_uploader("ارفع صورة MRI هنا", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # فتح الصورة
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="الصورة المرفوعة", use_container_width=True)

    # تحويل الصورة إلى NumPy
    img_np = np.array(pil_img)
    img_preprocessed = preprocess_images(img_np)

    # تجهيز عشان يدخل الموديل
    img_array = np.expand_dims(img_preprocessed, axis=0) / 255.0  

    # التنبؤ
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    # خريطة الكلاسات
    label_map = {0: "Normal", 1: "Alzheimer's", 2: "Parkinson's"}

    st.subheader("📌 النتيجة:")
    st.write(f"الموديل متوقع: **{label_map[class_idx]}**")

    # عرض نسب التوقع
    st.write("🔎 نسب التوقع:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{label_map[i]}: {prob:.2%}")
