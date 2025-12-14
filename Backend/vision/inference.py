# backend/vision/inference.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path

# =====================
# DEVICE
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# MODEL PATHS
# =====================
CHEST_MODEL_PATH = Path(__file__).resolve().parent / "models" / "chest_resnet18_multilabel.pth"
PNEUMONIA_MODEL_PATH = Path(__file__).resolve().parent / "models" / "pneumonia_resnet18_binary.pth"
BREAST_MODEL_PATH = Path(__file__).resolve().parent / "models" / "breast_resnet18_binary.pth"
ORGAN_MODEL_PATH = Path(__file__).resolve().parent / "models" / "organamnist_resnet18_multiclass.pth"

# =====================
# LABELS
# =====================
CHEST_LABELS = [
    "atelectasis", "cardiomegaly", "effusion", "infiltration",
    "mass", "nodule", "pneumonia", "pneumothorax",
    "consolidation", "edema", "emphysema", "fibrosis",
    "pleural_thickening", "hernia"
]

PNEUMONIA_LABELS = ["normal", "pneumonia"]
BREAST_LABELS = ["benign", "malignant"]

ORGAN_LABELS = [
    "bladder", "femur-left", "femur-right", "heart",
    "kidney-left", "kidney-right", "liver",
    "lung-left", "lung-right", "pancreas", "spleen"
]

# =====================
# IMAGE TRANSFORM  (MUST BE BEFORE FUNCTIONS)
# =====================
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# SAFE STATE_DICT LOADER
# =====================
def _load_state_dict_any(ckpt):
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt

# =====================
# LOADERS
# =====================
def load_chest_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 14)
    ckpt = torch.load(CHEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(_load_state_dict_any(ckpt))
    model.to(DEVICE)
    model.eval()
    return model

def load_pneumonia_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    ckpt = torch.load(PNEUMONIA_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(_load_state_dict_any(ckpt))
    model.to(DEVICE)
    model.eval()
    return model

def load_breast_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    ckpt = torch.load(BREAST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(_load_state_dict_any(ckpt))
    model.to(DEVICE)
    model.eval()
    return model

def load_organ_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 11)
    ckpt = torch.load(ORGAN_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(_load_state_dict_any(ckpt))
    model.to(DEVICE)
    model.eval()
    return model

# =====================
# CACHED MODELS
# =====================
_chest_model = None
_pneumonia_model = None
_breast_model = None
_organ_model = None

def get_chest_model():
    global _chest_model
    if _chest_model is None:
        _chest_model = load_chest_model()
    return _chest_model

def get_pneumonia_model():
    global _pneumonia_model
    if _pneumonia_model is None:
        _pneumonia_model = load_pneumonia_model()
    return _pneumonia_model

def get_breast_model():
    global _breast_model
    if _breast_model is None:
        _breast_model = load_breast_model()
    return _breast_model

def get_organ_model():
    global _organ_model
    if _organ_model is None:
        _organ_model = load_organ_model()
    return _organ_model

# =====================
# PREDICTIONS
# =====================
@torch.no_grad()
def predict_chest_from_pil(pil_img: Image.Image, threshold: float = 0.5):
    model = get_chest_model()
    pil_img = pil_img.convert("RGB")
    img = _transform(pil_img).unsqueeze(0).to(DEVICE)

    logits = model(img)
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    label_probs = sorted(
        [{"label": l, "prob": float(p)} for l, p in zip(CHEST_LABELS, probs)],
        key=lambda x: x["prob"],
        reverse=True
    )

    active = [lp for lp in label_probs if lp["prob"] >= threshold]

    return {
        "any_abnormal": len(active) > 0,
        "any_abnormal_prob": float(label_probs[0]["prob"]),
        "active_labels": active,
        "label_probs": label_probs
    }

@torch.no_grad()
def predict_pneumonia_from_pil(pil_img: Image.Image):
    model = get_pneumonia_model()
    pil_img = pil_img.convert("RGB")
    img = _transform(pil_img).unsqueeze(0).to(DEVICE)

    prob = torch.sigmoid(model(img)[0][0]).item()
    label = "pneumonia" if prob >= 0.5 else "normal"

    return {"predicted_label": label, "confidence": float(prob)}

@torch.no_grad()
def predict_breast_from_pil(pil_img: Image.Image):
    model = get_breast_model()
    pil_img = pil_img.convert("RGB")
    img = _transform(pil_img).unsqueeze(0).to(DEVICE)

    prob = torch.sigmoid(model(img)[0][0]).item()
    label = "malignant" if prob >= 0.5 else "benign"

    return {"predicted_label": label, "confidence": float(prob)}

@torch.no_grad()
def predict_organ_from_pil(pil_img: Image.Image):
    model = get_organ_model()
    pil_img = pil_img.convert("RGB")
    img = _transform(pil_img).unsqueeze(0).to(DEVICE)

    probs = torch.softmax(model(img), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))

    return {
        "predicted_label": ORGAN_LABELS[idx],
        "confidence": float(probs[idx])
    }
