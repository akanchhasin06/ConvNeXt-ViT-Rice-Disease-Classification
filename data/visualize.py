import matplotlib.pyplot as plt
import pandas as pd

from configs.config import TRAIN_CSV

df = pd.read_csv(TRAIN_CSV)

print("Total Images :", len(df))
print("Classes :", df["label"].unique())

plt.figure(figsize=(10,5))
df["label"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Disease")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()