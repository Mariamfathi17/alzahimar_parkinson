import os
import io
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as tvm
from PIL import Image
import gdown

st.set_page_config(page_title="Alzheimer’s vs Parkinson’s MRI Classifier", layout="centered")
DEVICE = torch.device("cpu")

# -----------------------------
# 1) Download model if missing
# -----------------------------
MODEL_URL = "https://drive.google.com/uc?id=15Kfi84AOr76Ul3o-jMdUZMXLQvL4LpRR"
MODEL_PATH = "alzheimers_parkinson_model.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model, please wait..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully.")

# -----------------------------
# 2) Utilities
# -----------------------------
def _strip_prefixes(state_dict, prefixes=("module.", "model.")):
    new_sd = {}
    for k, v in state_dict.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
        new_sd[k] = v
    return new_sd

def _infer_num_classes_from_state_dict(state_dict, default=3):
    # Try to find a final linear layer weight to infer out_features
    candidates = [k for k in state_dict.keys() if k.endswith("weight")]
    # Heuristic: prefer common classifier names
    order = ["fc.weight", "classifier.weight", "head.weight", "heads.weight", "last_linear.weight"]
    for name in order:
        if name in state_dict:
            w = state_dict[name]
            if isinstance(w, torch.Tensor) and w.ndim in (1, 2):
                return int(w.shape[0])
    # fallback: first linear-ish weight tensor with 2 dims
    for k in candidates:
        w = state_dict[k]
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[0])
    return default

def _build_resnet18(num_classes):
    model = tvm.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# -----------------------------
# 3) Load Model (cached)
# -----------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str) -> nn.Module:
    # Case A: TorchScript
    try:
        scripted = torch.jit.load(path, map_location=DEVICE)
        scripted.eval()
        return scripted
    except Exception:
        pass

    # Case B: torch.load -> could be Module, state_dict, or checkpoint
    obj = torch.load(path, map_location=DEVICE)

    # If it's a full module already
    if isinstance(obj, nn.Module):
        obj.to(DEVICE)
        obj.eval()
        return obj

    # If it's a dict: either plain state_dict or checkpoint with 'state_dict'
    if isinstance(obj, dict):
        state_dict = obj.get("state_dict", obj)
        # Ensure it’s a real state_dict mapping to tensors
        if not any(isinstance(v, torch.Tensor) for v in state_dict.values()):
            raise ValueError("Loaded dict does not look like a PyTorch state_dict.")
        state_dict = _strip_prefixes(state_dict)

        num_classes = _infer_num_classes_from_state_dict(state_dict, default=3)
        model = _build_resnet18(num_classes=num_classes)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Not fatal, but useful to surface:
        if missing or unexpected:
            st.info(
                f"Loaded with non-strict keys. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
            )
        model.to(DEVICE)
        model.eval()
        return model

    # Unknown format
    raise TypeError(
        f"Unsupported model file type: {type(obj)}. "
        "Please save either a TorchScript model, a full nn.Module, or a state_dict/checkpoint."
    )

model = load_model(MODEL_PATH)

# -----------------------------
# 4) Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# 5) UI
# -----------------------------
st.title("Alzheimer’s vs Parkinson’s MRI Classifier")
st.markdown("Upload an MRI image to predict **Normal**, **Alzheimer’s**, or **Parkinson’s**.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

label_map = {
    0: "Normal",
    1: "Alzheimer’s",
    2: "Parkinson’s",
}

if uploaded_file:
    # Read image
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    # Inference
    with torch.inference_mode():
        x = transform(image).unsqueeze(0).to(DEVICE)
        outputs = model(x)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        if not isinstance(outputs, torch.Tensor):
            raise RuntimeError("Model forward did not return a tensor of logits.")

        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_label = label_map.get(pred_idx, f"Class {pred_idx}")

    st.subheader("Prediction Result")
    st.success(f"→ **{pred_label}**")

    st.subheader("Class Probabilities")
    for idx, name in label_map.items():
        if idx < len(probs):
            st.write(f"- {name}: {probs[idx]*100:.2f}%")

    # Simple bar chart
    data = {label_map[i]: float(probs[i]) for i in range(min(len(probs), len(label_map)))}
    st.bar_chart(data)
