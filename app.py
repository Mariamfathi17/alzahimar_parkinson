import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# ------------------------------
# تحميل الموديل (Keras)
# ------------------------------
model = tf.keras.models.load_model("mlp_model_final.keras")

# نطبع شكل المدخلات عشان نفهم هو عايز إيه
st.write("✅ Model input shape:", model.input_shape)

# ------------------------------
# دالة الـ preprocessing
# ------------------------------
def preprocess_image(image, model_input_shape):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))  
    img = img / 255.0

    # لو الموديل بياخد (None, 224,224,3) → CNN
    if len(model_input_shape) == 4:
        img = img.reshape(1, 224, 224, 3)
    # لو الموديل بياخد (None, 150528) → MLP
    else:
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

    # معالجة الصورة حسب الموديل
    processed_img = preprocess_image(image, model.input_shape)

    # توقع النتيجة
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # الماب بتاع الكلاسات
    label_map = {0: "Alzheimer", 1: "Normal", 2: "Parkinson"}

    # عرض النتيجة
    st.subheader("Prediction Result:")
    st.write(f"📌 **Class:** {label_map[predicted_class]}")
    st.write(f"🔢 Raw Probabilities: {prediction}")
