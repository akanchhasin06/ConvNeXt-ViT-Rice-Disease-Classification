import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from configs.config import *

from data.dataset import RiceDiseaseDataset
from data.transforms import (
    get_train_transforms,
    get_valid_transforms
)

from models import build_model

from trainer.trainer import Trainer

from utils.split_data import create_train_valid_split
from utils.seed import set_seed


def main():

    print("=" * 60)
    print("Rice Disease Classification Training")
    print("=" * 60)

    # --------------------------------------------------
    # Set Seed
    # --------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------
    # Read Dataset
    # --------------------------------------------------

    df = pd.read_csv(TRAIN_CSV)

    classes = sorted(df["label"].unique())

    class_to_idx = {
        cls: idx
        for idx, cls in enumerate(classes)
    }

    print("\nClasses Found:")
    print(class_to_idx)

    # --------------------------------------------------
    # Train / Validation Split
    # --------------------------------------------------

    train_df, valid_df = create_train_valid_split(df)

    train_dataset = RiceDiseaseDataset(
        dataframe=train_df,
        image_dir=TRAIN_DIR,
        class_to_idx=class_to_idx,
        transform=get_train_transforms()
    )

    valid_dataset = RiceDiseaseDataset(
        dataframe=valid_df,
        image_dir=TRAIN_DIR,
        class_to_idx=class_to_idx,
        transform=get_valid_transforms()
    )

    print(f"\nTraining Samples   : {len(train_dataset)}")
    print(f"Validation Samples : {len(valid_dataset)}")

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------


    model = build_model(
        MODEL,
        NUM_CLASSES
    ).to(DEVICE)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

# --------------------------------------------------
# Optimizer
# --------------------------------------------------

    if MODEL == "hybrid":

        optimizer = optim.AdamW(
            [
                {
                    "params": model.convnext.parameters(),
                    "lr": 1e-5
                },
                {
                    "params": model.vit.parameters(),
                    "lr": 1e-5
                },
                {
                    "params":
                        list(model.conv_projection.parameters()) +
                        list(model.vit_projection.parameters()) +
                        list(model.cross_attention.parameters()) +
                        list(model.fusion.parameters()) +
                        list(model.classifier.parameters()),
                    "lr": 1e-4
                }
            ],
            weight_decay=WEIGHT_DECAY
        )

    elif MODEL == "convnext":

        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=WEIGHT_DECAY
        )

    elif MODEL == "vit":

        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=WEIGHT_DECAY
        )

    else:

        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )   
        
    
    
    # --------------------------------------------------
    # Scheduler
    # --------------------------------------------------

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=MIN_LR
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE
    )

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------

    best_val_acc = 0.0

    print("Checkpoint will be saved to:")
    print(CHECKPOINT_NAME)

    print("\nStarting Training...\n")

    for epoch in range(NUM_EPOCHS):

        train_loss, train_acc = trainer.train_one_epoch()

        val_loss, val_acc = trainer.validate()

        scheduler.step(val_loss)

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                CHECKPOINT_NAME
            )

            print("✅ Best model saved!")

        print("-" * 60)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")

        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")

        print(f"Best Val Acc : {best_val_acc:.4f}")

    print("\nTraining Completed Successfully!")

    print(f"\nBest Validation Accuracy : {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

