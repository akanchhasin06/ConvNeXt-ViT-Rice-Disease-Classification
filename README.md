# ConvNeXt-ViT Hybrid Network with Cross-Attention for Rice Disease Classification

## Overview

This project proposes a hybrid deep learning architecture combining ConvNeXt Tiny and Vision Transformer (ViT) using a Cross-Attention Fusion module for rice disease classification.

The objective is to improve classification accuracy by leveraging both convolutional local features and transformer-based global representations.

---

## Features

- ConvNeXt Tiny Encoder
- Vision Transformer Encoder
- Cross-Attention Feature Fusion
- Fine-tuning using pretrained backbones
- Multi-class rice disease classification
- Evaluation with confusion matrix and ROC curves

---

## Dataset

Rice Disease Dataset

Classes:

- bacterial_leaf_blight
- bacterial_leaf_streak
- bacterial_panicle_blight
- blast
- brown_spot
- dead_heart
- downy_mildew
- hispa
- normal
- tungro

Total Images: 10,407

---

## Project Structure

```text
configs/
data/
dataset/
models/
trainer/
utils/

checkpoints/
outputs/

train.py
evaluate.py
predict.py
```

---

## Results

| Model | Validation Accuracy |
|--------|--------------------:|
| CNN | 47.02% |
| ConvNeXt | 77.86% |
| ViT | 76.90% |
| Proposed Hybrid | **97.69%** |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Train

```bash
python train.py
```

---

## Evaluate

```bash
python evaluate.py
```

---

## Predict

```bash
python predict.py
```

---

## Author

Akanchha Singh
B.Tech CSE
Graphic Era University