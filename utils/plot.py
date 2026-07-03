import matplotlib.pyplot as plt


def plot_training_curves(
    train_loss,
    val_loss,
    train_acc,
    val_acc,
    save_dir
):

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8,6))

    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")

    plt.legend()

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.savefig(save_dir / "loss_curve.png")

    plt.close()

    plt.figure(figsize=(8,6))

    plt.plot(epochs, train_acc, label="Train Accuracy")

    plt.plot(epochs, val_acc, label="Validation Accuracy")

    plt.legend()

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.savefig(save_dir / "accuracy_curve.png")

    plt.close()