import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.title("Alzheimer MRI Prediction with Enhanced Saliency Map")

# -------- 1) Load the model --------
@st.cache_resource
def load_model_streamlit():
    return load_model("alzheimer_cnn_model.h5")

model = load_model_streamlit()

class_labels = ["AD", "MCI", "CN"]

# -------- 2) Upload image --------
uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((128,128))
    x = np.array(img_resized)/255.0
    x = np.expand_dims(x, axis=0)

    # -------- 3) Prediction --------
    preds = model.predict(x)
    predicted_class = np.argmax(preds[0])
    confidence = preds[0][predicted_class]

    # -------- 4) Saliency Map --------
    img_tensor = tf.convert_to_tensor(x)
    img_tensor = tf.cast(img_tensor, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        loss = predictions[:, predicted_class]
    grads = tape.gradient(loss, img_tensor)[0]
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency)-tf.reduce_min(saliency))
    saliency_uint8 = np.uint8(255 * saliency)

    # -------- 5) Streamlit sliders --------
    gauss_ksize = st.slider("Gaussian Kernel Size (odd)", 1, 21, 7, step=2)
    clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.1)
    sharpen_strength = st.slider("Sharpen Strength", 0.5, 3.0, 1.0, 0.1)
    overlay_alpha = st.slider("Overlay Alpha", 0.1, 1.0, 0.4, 0.05)

    # -------- 6) Apply filters --------
    # Gaussian Blur
    saliency_blur = cv2.GaussianBlur(saliency_uint8, (gauss_ksize, gauss_ksize), 0)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
    saliency_clahe = clahe.apply(saliency_blur)
    # Sharpen
    kernel_sharp = np.array([[0,-1,0], [-1,5*sharpen_strength,-1], [0,-1,0]])
    saliency_sharp = cv2.filter2D(saliency_clahe, -1, kernel_sharp)
    # Color Map
    saliency_color = cv2.applyColorMap(saliency_sharp, cv2.COLORMAP_JET)
    # Overlay
    original_img_cv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
    overlay_img = cv2.addWeighted(original_img_cv, 1-overlay_alpha, saliency_color, overlay_alpha, 0)

    # -------- 7) Display images --------
    st.subheader("Original Image vs Enhanced Saliency Overlay")
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(np.array(img_resized))
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"Enhanced Saliency Overlay - {class_labels[predicted_class]}")
    ax[1].axis("off")
    st.pyplot(fig)

    # -------- 8) Prediction Probabilities --------
    st.subheader("Prediction Probabilities")
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.barplot(x=class_labels, y=preds[0], ax=ax2)
    ax2.set_ylim(0,1)
    st.pyplot(fig2)

    # -------- 9) Text report --------
    st.subheader("Prediction Report")
    st.write(f"Predicted Class: **{class_labels[predicted_class]}**")
    st.write(f"Confidence: **{confidence:.2f}**")
    st.write("Full Probabilities:")
    for i, label in enumerate(class_labels):
        st.write(f"{label}: {preds[0][i]:.2f}")
