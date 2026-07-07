import matplotlib.pyplot as plt

# -----------------------------
# Model Comparison
# -----------------------------

models = [
    "CNN",
    "ConvNeXt",
    "ViT",
    "Hybrid"
]

accuracy = [
    47.02,
    77.86,
    76.90,
    97.69
]

plt.figure(figsize=(8,6))

bars = plt.bar(models, accuracy)

plt.ylim(0,100)

plt.ylabel("Validation Accuracy (%)")

plt.title("Performance Comparison of Different Models")

for bar in bars:

    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+1,
        f"{bar.get_height():.2f}",
        ha="center",
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison.png",
    dpi=300
)

plt.close()

print("Model comparison plot saved!")