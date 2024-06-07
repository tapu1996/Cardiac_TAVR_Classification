from typing import Tuple, Any

import torch
import torch.nn as nn
from classeg.training.trainer import Trainer, log
import torchvision.transforms as transforms
from classeg.utils.constants import PREPROCESSED_ROOT
from classeg.utils.utils import read_json
from tqdm import tqdm


class ClassificationTrainer(Trainer):
    def __init__(self, dataset_name: str, fold: int, model_path: str, gpu_id: int, unique_folder_name: str,
                 config_name: str, resume: bool = False, preload: bool = True, world_size: int = 1):
        """
        Trainer class for training and checkpointing of networks.
        :param dataset_name: The name of the dataset to use.
        :param fold: The fold in the dataset to use.
        :param save_latest: If we should save a checkpoint each epoch
        :param model_path: The path to the json that defines the architecture.
        :param gpu_id: The gpu for this process to use.
        :param checkpoint_name: None if we should train from scratch, otherwise the model weights that should be used.
        """
        super().__init__(dataset_name, fold, model_path, gpu_id, unique_folder_name, config_name, resume, preload,
                         world_size)

        class_names = read_json(f"{PREPROCESSED_ROOT}/{self.dataset_name}/id_to_label.json")
        self.class_names = [i for i in sorted(class_names.values())]
        self._last_val_accuracy = 0.
        self._val_accuracy = 0.
        self.softmax = nn.Softmax(dim=1)

    def get_augmentations(self) -> Tuple[Any, Any]:
        # train_aug = transforms.Compose([
        #     transforms.Resize(self.config.get('target_size', [512, 512]), antialias=True),
        #     transforms.RandomRotation(degrees=30),
        #     transforms.RandomAdjustSharpness(1.3),
        #     transforms.RandomVerticalFlip(p=0.25),
        #     transforms.RandomHorizontalFlip(p=0.25),
        #     transforms.RandomErasing(p=0.25, scale=(0.02, 0.33))
        # ])
        # val_aug = transforms.Compose([
        #     transforms.Resize(self.config.get('target_size', [512, 512]), antialias=True)
        # ])

        return None, None

    def train_single_epoch(self, epoch) -> float:
        """
        The training of each epoch is done here.
        :return: The mean loss of the epoch.
        """
        running_loss = 0.
        total_items = 0
        # ForkedPdb().set_trace()
        log_image = epoch % 10 == 0
        for data, labels, _ in tqdm(self.train_dataloader):
            self.optim.zero_grad()
            if log_image:
                self.log_helper.log_augmented_image(data[0][0][100])
            labels = labels.to(self.device, non_blocking=True)
            data = data.to(self.device)
            batch_size = data.shape[0]
            # ForkedPdb().set_trace()
            # do prediction and calculate loss
            predictions = self.model(data)
            loss = self.loss(predictions, labels)
            # update model
            loss.backward()
            self.optim.step()
            # gather data
            running_loss += loss.item() * batch_size
            total_items += batch_size

        return running_loss / total_items

    def post_epoch_log(self, epoch: int) -> Tuple:
        """
        Executed after each default logging cycle
        """
        message = f"Val accuracy: {self._val_accuracy} --change-- {self._val_accuracy - self._last_val_accuracy}"
        self._last_val_accuracy = self._val_accuracy
        return message,

    # noinspection PyTypeChecker
    def eval_single_epoch(self, epoch) -> float:
        """
        Runs evaluation for a single epoch.
        :return: The mean loss and mean accuracy respectively.
        """

        running_loss = 0.
        correct_count = 0.
        total_items = 0
        all_predictions, all_labels = [], []
        for data, labels, _ in tqdm(self.val_dataloader):
            labels = labels.to(self.device, non_blocking=True)
            data = data.to(self.device)
            batch_size = data.shape[0]
            # do prediction and calculate loss
            predictions = self.model(data)
            loss = self.loss(predictions, labels)
            running_loss += loss.item() * batch_size
            # analyze
            predictions = torch.argmax(self.softmax(predictions), dim=1)
            # labels = torch.argmax(labels, dim=1)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            correct_count += torch.sum(predictions == labels)
            total_items += batch_size
        self.log_helper.plot_confusion_matrix(all_predictions, all_labels, self.class_names)
        self._val_accuracy = correct_count / total_items
        return running_loss / total_items

    def get_loss(self) -> nn.Module:
        """
        Build the criterion object.
        :return: The loss function to be used.
        """
        if self.device == 0:
            log("Loss being used is nn.CrossEntropyLoss()")
        return nn.CrossEntropyLoss()

    def get_model(self, _: str) -> nn.Module:
        """
        Build the model object.
        :return: The model to be used.
        """
        from classeg.extensions.tavr.training.model import ClassNet
        return ClassNet(in_channels=10, out_channels=2)
