import torch
from pathlib import Path

# -----------------------------
# Project Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_PATH / "train_images"
TEST_DIR = DATASET_PATH / "test_images"

TRAIN_CSV = DATASET_PATH / "train.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# -----------------------------
# Image Parameters
# -----------------------------
IMAGE_SIZE = 224

# -----------------------------
# Training Parameters
# -----------------------------
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

# -----------------------------
# Hardware
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 2

# -----------------------------
# Random Seed
# -----------------------------
SEED = 42