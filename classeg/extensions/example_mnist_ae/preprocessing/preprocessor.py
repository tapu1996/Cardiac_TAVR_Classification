import shutil
from typing import Dict

import tqdm
import numpy as np
from overrides import override
from torchvision.datasets import MNIST

from classeg.preprocessing.preprocessor import Preprocessor
from classeg.utils.constants import RAW_ROOT, DEFAULT_PROCESSES
from classeg.utils.utils import get_case_name_from_number, check_raw_exists


class MnistPreprocessor(Preprocessor):
    def __init__(self, dataset_id: str, folds: int, processes: int, normalize: bool, **kwargs):
        super().__init__(dataset_id, folds, processes, normalize, **kwargs)
        self.normalize = True

    @override
    def get_config(self) -> Dict:
        return {
            "batch_size": 64,
            "processes": DEFAULT_PROCESSES,
            "lr": 0.001,
            "epochs": 50,
            "momentum": 0,
            "weight_decay": 0.0001,
            "target_size": [24, 24]
        }

    @override
    def pre_preprocessing(self):
        check_raw_exists(self.dataset_name)
        dataset = MNIST(".", train=True, download=True)
        case_number = 0
        for image, _ in tqdm.tqdm(dataset, desc="Downloading MNIST"):
            image.save(f"{RAW_ROOT}/{self.dataset_name}/{get_case_name_from_number(case_number)}.jpg")
            case_number += 1
        shutil.rmtree("./MNIST")

    @override
    def normalize_function(self, data: np.array) -> np.array:
        """
        Perform normalization. z-score normalization will still always occur for classification and segmentation
        """
        return data / 255
