import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# تحميل الموديل keras
model = tf.keras.models.load_model("cnn_model.keras")

# عنوان الموقع
st.title("🧠 Alzheimer's / Parkinson's / Normal MRI Classifier")

# رفع الصورة من المستخدم
uploaded_file = st.file_uploader("ارفع صورة MRI هنا", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة
    img = Image.open(uploaded_file)
    st.image(img, caption="الصورة المرفوعة", use_column_width=True)

    # تجهيز الصورة عشان تدخل الموديل
    img = img.resize((224, 224))  # لازم تبقى نفس المقاس اللي اتدرب عليه الموديل
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # تطبيع القيم

    # التنبؤ
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    # خريطة الكلاسات
    label_map = {0: "Normal", 1: "Alzheimer's", 2: "Parkinson's"}

    st.subheader("📌 النتيجة:")
    st.write(f"الموديل متوقع: **{label_map[class_idx]}**")

    # لو حبيتي تشوفي النسب كمان
    st.write("🔎 نسب التوقع:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{label_map[i]}: {prob:.2%}")
