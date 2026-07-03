from .cnn import BaselineCNN
from .convnext_classifier import ConvNeXtClassifier
from .vit_classifier import ViTClassifier
from .hybrid import HybridModel


def build_model(model_name, num_classes):

    if model_name == "cnn":
        return BaselineCNN(num_classes)

    elif model_name == "convnext":
        return ConvNeXtClassifier(num_classes)

    elif model_name == "vit":
        return ViTClassifier(num_classes)

    elif model_name == "hybrid":
        return HybridModel(num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")