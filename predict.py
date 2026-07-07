import torch
import pandas as pd
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.config import *
from models import build_model


# ----------------------------
# Image Transform
# ----------------------------

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])


# ----------------------------
# Load Classes
# ----------------------------

df = pd.read_csv(TRAIN_CSV)

classes = sorted(df["label"].unique())

idx_to_class = {
    idx: cls
    for idx, cls in enumerate(classes)
}


# ----------------------------
# Load Model
# ----------------------------

model = build_model(
    MODEL,
    NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        CHECKPOINT_NAME,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()


# ----------------------------
# Prediction Function
# ----------------------------

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = np.array(image)

    image = transform(image=image)["image"]

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

    # -------------------------
    # Top 3 Predictions
    # -------------------------

    top_probs, top_indices = torch.topk(
        probabilities,
        k=3,
        dim=1
    )

    print("\n" + "=" * 55)
    print("Rice Disease Prediction")
    print("=" * 55)

    for i in range(3):

        disease = idx_to_class[
            top_indices[0][i].item()
        ]

        confidence = (
            top_probs[0][i].item() * 100
        )

        print(
            f"{i+1}. {disease:<30} {confidence:.2f}%"
        )

    print("-" * 55)

    final_confidence = top_probs[0][0].item() * 100

    if final_confidence >= 95:

        level = "High Confidence"

    elif final_confidence >= 80:

        level = "Moderate Confidence"

    else:

        level = "Low Confidence"

    print(f"\nFinal Prediction : {idx_to_class[top_indices[0][0].item()]}")
    print(f"Confidence       : {final_confidence:.2f}%")
    print(f"Prediction Level : {level}")

    return idx_to_class[top_indices[0][0].item()]

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    image_path = input("\nEnter Image Path: ").strip()

    predict(image_path)