import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class RiceDiseaseDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        class_to_idx,
        transform=None
    ):

        self.data = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

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
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        label = self.class_to_idx[label_name]

        return image, label