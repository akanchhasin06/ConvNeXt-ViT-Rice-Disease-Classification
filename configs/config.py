import torch
from pathlib import Path

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATASET_PATH / "train_images"

TEST_DIR = DATASET_PATH / "test_images"

TRAIN_CSV = DATASET_PATH / "train.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================================
# Image Parameters
# =====================================================

IMAGE_SIZE = 224
NUM_CLASSES = 10

# =====================================================
# Training Parameters
# =====================================================

BATCH_SIZE = 16
NUM_EPOCHS = 25

# Fine-tuning learning rate
LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

# =====================================================
# Scheduler
# =====================================================

PATIENCE = 5
FACTOR = 0.5
MIN_LR = 1e-6

# =====================================================
# Hardware
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

# =====================================================
# Model
# =====================================================
MODEL = "hybrid"

MODEL_NAME = "HybridModel"

CHECKPOINT_NAME = CHECKPOINT_DIR / f"{MODEL}.pth"

# =====================================================
# Random Seed
# =====================================================

SEED = 42