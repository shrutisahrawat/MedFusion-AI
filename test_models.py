from PIL import Image
import numpy as np

from Backend.vision.inference import (
    predict_chest_from_pil,
    predict_pneumonia_from_pil,
    predict_breast_from_pil,
    predict_organ_from_pil
)

# Create a dummy RGB image (224x224)
dummy_img = Image.fromarray(
    (np.random.rand(224, 224, 3) * 255).astype("uint8")
)

print("CHEST:", predict_chest_from_pil(dummy_img))
print("PNEUMONIA:", predict_pneumonia_from_pil(dummy_img))
print("BREAST:", predict_breast_from_pil(dummy_img))
print("ORGAN:", predict_organ_from_pil(dummy_img))
