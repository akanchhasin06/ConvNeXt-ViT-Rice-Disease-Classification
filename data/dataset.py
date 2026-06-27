import os
import pandas as pd
from PIL import Image
import numpy as np

from torch.utils.data import Dataset


class RiceDiseaseDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):

        self.data = pd.read_csv(csv_file)

        self.image_dir = image_dir

        self.transform = transform

        # Create label mapping
        self.classes = sorted(self.data["label"].unique())

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.classes)
        }

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_name = row["image_id"]

        label_name = row["label"]

        image_path = os.path.join(
            self.image_dir,
            label_name,
            image_name
        )

        image = Image.open(image_path).convert("RGB")

        label = self.class_to_idx[label_name]
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label