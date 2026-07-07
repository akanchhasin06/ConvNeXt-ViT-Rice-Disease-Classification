import pandas as pd
import matplotlib.pyplot as plt

history = pd.read_csv("outputs/training_history.csv")

# ---------------- Accuracy ----------------

plt.figure(figsize=(8,6))

plt.plot(
    history["Epoch"],
    history["Train Acc"] * 100,
    linewidth=2,
    label="Train Accuracy"
)

plt.plot(
    history["Epoch"],
    history["Val Acc"] * 100,
    linewidth=2,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")
plt.grid(True)
plt.legend()

plt.savefig(
    "outputs/accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------- Loss ----------------

plt.figure(figsize=(8,6))

plt.plot(
    history["Epoch"],
    history["Train Loss"],
    linewidth=2,
    label="Train Loss"
)

plt.plot(
    history["Epoch"],
    history["Val Loss"],
    linewidth=2,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.legend()

plt.savefig(
    "outputs/loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Training curves saved successfully!")