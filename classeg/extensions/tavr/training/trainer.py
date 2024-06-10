from typing import Tuple, Any

import torch
import torch.nn as nn
from classeg.training.trainer import Trainer, log
from torch.optim import SGD
import torchvision.transforms as transforms
from classeg.utils.constants import PREPROCESSED_ROOT
from classeg.utils.utils import read_json
from tqdm import tqdm
from monai.transforms import Compose, CenterSpatialCrop, Lambda
from classeg.utils.utils import get_dataloaders_from_fold
from torch.utils.data import WeightedRandomSampler, DistributedSampler

class ClassificationTrainer(Trainer):
    def __init__(self, dataset_name: str, fold: int, model_path: str, gpu_id: int, unique_folder_name: str,
                 config_name: str, resume: bool = False, cache: bool = True, world_size: int = 1):
        """
        Trainer class for training and checkpointing of networks.
        :param dataset_name: The name of the dataset to use.
        :param fold: The fold in the dataset to use.
        :param save_latest: If we should save a checkpoint each epoch
        :param model_path: The path to the json that defines the architecture.
        :param gpu_id: The gpu for this process to use.
        :param checkpoint_name: None if we should train from scratch, otherwise the model weights that should be used.
        """
        super().__init__(dataset_name, fold, model_path, gpu_id, unique_folder_name, config_name, resume, cache,
                         world_size)

        class_names = read_json(f"{PREPROCESSED_ROOT}/{self.dataset_name}/id_to_label.json")
        self.class_names = [i for i in sorted(class_names.values())]
        self._last_val_accuracy = 0.
        self._last_train_accuracy = 0.
        self._val_accuracy = 0.
        self._train_accuracy = 0.
        self.softmax = nn.Softmax(dim=1)


    def get_dataloaders(self):
        """
        This method is responsible for creating the augmentation and then fetching dataloaders.

        :return: Train and val dataloaders.
        """
        train_transforms, val_transforms = self.get_augmentations()
        return get_dataloaders_from_fold(
            self.dataset_name,
            self.fold,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            sampler=(WeightedRandomSampler if self.world_size == 1 else DistributedSampler),
            cache=self.cache,
            rank=self.device,
            world_size=self.world_size,
            config_name=self.config_name
        )

    def get_augmentations(self) -> Tuple[Any, Any]:
        def binarize(x):
            bg = x[0, 0, 0, 0]
            x[x==bg] = 0
            x[x!=0]=1
            return x
        
        train = Compose([
            Lambda(func=lambda x:binarize(x)),
            CenterSpatialCrop(roi_size=self.config["target_size"])
        ])

        return train, train

    def train_single_epoch(self, epoch) -> float:
        """
        The training of each epoch is done here.
        :return: The mean loss of the epoch.
        """
        running_loss = 0.
        total_items = 0
        correct_count = 0.
        all_predictions, all_labels = [], []
        # ForkedPdb().set_trace()
        log_image = True
        i = 0
        for data, labels, _ in tqdm(self.train_dataloader):
            i += 1
            self.optim.zero_grad()
            if log_image and i == 1:
                self.log_helper.log_augmented_image(data[0][0][100].unsqueeze(0))
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
            predictions = torch.argmax(self.softmax(predictions), dim=1)
            correct_count += torch.sum(predictions == labels)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            # gather data
            running_loss += loss.item() * batch_size
            total_items += batch_size

        self.log_helper.plot_confusion_matrix(all_predictions, all_labels, self.class_names, set_name="train")
        self._train_accuracy = correct_count / total_items
        return running_loss / total_items

    def post_epoch_log(self, epoch: int) -> Tuple:
        """
        Executed after each default logging cycle
        """
        messageval = f"Val accuracy: {self._val_accuracy} --change-- {self._val_accuracy - self._last_val_accuracy}"
        messagetrain = f"Train accuracy: {self._train_accuracy} --change-- {self._train_accuracy - self._last_train_accuracy}"
        self._last_val_accuracy = self._val_accuracy
        self._last_train_accuracy = self._train_accuracy
        return messageval, messagetrain

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
        i = 0
        for data, labels, _ in tqdm(self.val_dataloader):
            i += 1
            labels = labels.to(self.device, non_blocking=True)
            data = data.to(self.device)
            if i == 1 and epoch%10 == 0:
                self.log_helper.log_net_structure(self.model, data)
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
    
    def get_optim(self) -> Any:
        return SGD(
            self.model.parameters(),
            lr=self.config.get("lr"),
            momentum=self.config["momentum"],
            weight_decay=self.config["weight_decay"]
        )

    def get_model(self, name: str) -> nn.Module:
        """
        Build the model object.
        :return: The model to be used.
        """
        if name in ["ClassNet", "classnet", "cn"]:
            from classeg.extensions.tavr.training.model import ClassNet
            return ClassNet(in_channels=2, out_channels=2).to(self.device)
        elif name in ["embed", "ClassNetEmbed", "ClassNetEmbedding", "cne"]:
            from classeg.extensions.tavr.training.embed_model import ClassNetEmbedding
            return ClassNetEmbedding(in_channels=1, out_channels=2).to(self.device)
        elif name in ["lstm", "ClassNetLstm", "cnl"]:
            from classeg.extensions.tavr.training.lstm_model import ClassNetLSTM
            return ClassNetLSTM(in_channels=1, out_channels=2).to(self.device)
        