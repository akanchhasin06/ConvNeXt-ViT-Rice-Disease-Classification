import torch
from tqdm import tqdm

from utils.metrics import calculate_accuracy, AverageMeter


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        device,
    ):

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self):

        self.model.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        progress = tqdm(self.train_loader)

        for images, labels in progress:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            accuracy = calculate_accuracy(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(accuracy, images.size(0))

            progress.set_description(
                f"Loss:{loss_meter.avg:.4f} Acc:{acc_meter.avg:.4f}"
            )

        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for images, labels in self.valid_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            accuracy = calculate_accuracy(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(accuracy, images.size(0))

        return loss_meter.avg, acc_meter.avg