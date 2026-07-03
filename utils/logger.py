import os
import csv


class CSVLogger:

    def __init__(self, filepath):

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self.filepath = filepath

        with open(filepath, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "Epoch",
                "Train Loss",
                "Train Accuracy",
                "Validation Loss",
                "Validation Accuracy"
            ])

    def log(
        self,
        epoch,
        train_loss,
        train_acc,
        val_loss,
        val_acc
    ):

        with open(self.filepath, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc
            ])