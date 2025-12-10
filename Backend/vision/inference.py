# backend/vision/inference.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(__file__).resolve().parent / "models" / "chest_resnet18_multilabel.pth"

# ChestMNIST official labels (14)
CHEST_LABELS = [
    "atelectasis", "cardiomegaly", "effusion", "infiltration",
    "mass", "nodule", "pneumonia", "pneumothorax",
    "consolidation", "edema", "emphysema", "fibrosis",
    "pleural_thickening", "hernia"
]

# Image preprocess (same as training)
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_chest_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 14)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

_chest_model = None

def get_chest_model():
    global _chest_model
    if _chest_model is None:
        _chest_model = load_chest_model()
    return _chest_model


@torch.no_grad()
def predict_chest_from_pil(pil_img: Image.Image, threshold: float = 0.5):
    model = get_chest_model()

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    img = _transform(pil_img).unsqueeze(0).to(DEVICE)

    logits = model(img)
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    label_probs = []
    for lbl, p in zip(CHEST_LABELS, probs):
        label_probs.append({
            "label": lbl,
            "prob": float(p)
        })

    # sort by probability
    label_probs = sorted(label_probs, key=lambda x: x["prob"], reverse=True)

    active_labels = [lp for lp in label_probs if lp["prob"] >= threshold]

    result = {
        "any_abnormal": len(active_labels) > 0,
        "any_abnormal_prob": float(label_probs[0]["prob"]),
        "active_labels": active_labels,
        "label_probs": label_probs,
    }
    return result
