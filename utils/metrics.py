import torch


def calculate_accuracy(outputs, labels):
    """
    Calculates classification accuracy.
    """
    _, predictions = torch.max(outputs, dim=1)

    correct = (predictions == labels).sum().item()

    total = labels.size(0)

    accuracy = correct / total

    return accuracy


class AverageMeter:
    """
    Keeps track of average values during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count