import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# تحميل الموديل
model = load_model("cnn_model.keras")

# رفع الصورة
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # قراءة الصورة
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # نخليها ملونة 3 قنوات
    
    # Resize نفس مقاس التدريب (هنا 224x224 افتراضي، غيرها لو مختلف عندك)
    img = cv2.resize(img, (224, 224))
    
    # Normalization (0 → 1)
    img = img.astype("float32") / 255.0
    
    # إضافة batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img, axis=0)

    # التنبؤ
    prediction = model.predict(img_array)

    st.write("Prediction:", prediction)
