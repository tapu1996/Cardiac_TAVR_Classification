from typing import Dict

import numpy as np
from PIL import ImageFile
from overrides import override

from classeg.preprocessing.preprocessor import Preprocessor
from classeg.preprocessing.splitting import Splitter
from classeg.utils.constants import *
import pandas as pd
from classeg.utils.utils import get_case_name_from_number

ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
Extensions require to keep class name the same for proper loading
"""


class ExtensionPreprocessor(Preprocessor):
    def __init__(self, dataset_id: str, folds: int, processes: int, normalize: bool, dataset_desc: str = None, 
    ids_mapping: str = None, path_to_lv: str = None, **kwargs):
        """
        :param folds: How many folds to generate.
        :param processes: How many processes should be used.
        :param normalize: Should normalized data be saved.
        :param dataset_id: The id of the dataset.

        This is the main driver for preprocessing.
        """
        super().__init__(dataset_id, folds, processes, normalize, dataset_desc, **kwargs)
        if ids_mapping is None:
            raise SystemExit("Pass ids_mapping=path to csv")
        if path_to_lv is None:
            raise SystemExit("Pass path_to_lv=path to path_to_lv")
        self.ids_mapping = pd.from_csv(ids_mapping)
        self.path_to_lv = path_to_lv
        self.mode: Union["images", "segmentations"] = "images"
        if self.mode == "segmentations":
            # self.skip_zscore_norm = True
            ...

    def get_config(self) -> Dict:
        return {
            "batch_size": 32,
            "processes": DEFAULT_PROCESSES,
            "lr": 0.001,
            "epochs": 50,
            "momentum": 0,
            "weight_decay": 0.0001,
            "target_size": [224, 224, 224]
        }

    def normalize_function(self, data: np.array) -> np.array:
        """
        Perform normalization. z-score normalization will still always occur for classification and segmentation
        """
        return data

    def post_preprocessing(self):
        """
        Called at the end of preprocessing
        """
        ...

    @override
    def pre_preprocessing(self):
        """
        Called before standard preprocessing flow
        """
        # setup the raw structure -> stacking stuff -> class folders
        raw_dir = f"{RAW_ROOT}/{self.dataset_name}/"
        os.makedirs(raw_dir, exists_ok=True)
        existing_samples = glob.glob(f"{self.path_to_lv}/{self.mode}/*.nii.gz")
        for i, row in self.ids_mapping.iterrows():
            end_p = row["End Point"]
            set_type: Union["Train", "Test"] = row["Dataset_type"]
            if set_type == "Test":
                continue
            raw_folder = row["folder"]
            point_id = row["id"]
            # find the nii for the things and stack and move
            target_directory = f"{raw_dir}/{end_p}"
            os.makedirs(target_directory, exists_ok=True)
            # load
            case_name = get_case_name_from_number(int(point_id)) # case_xxxxx
            samples_for_row = [x for x in existing_samples if f"_{case_name.split(_)[1]}_0000_LV" in x]
            samples_for_row = sorted(samples_for_row)
            samples_for_row = [sitk.ReadImage(x) for x in samples_for_row]
            metadata = {
                "spacing": samples_for_row[0].GetSpacing(),
                "origin": samples_for_row[0].GetOrigin(),
                "direction": samples_for_row[0].GetDirection()
            }
            samples_for_row = np.stack([sitk.GetArrayFromImage(x) for x in samples_for_row])
            final_sample = sitk.GetImageFromArray(samples_for_row)
            final_sample.SetDirection(metadata["direction"])
            final_sample.SetOrigin(metadata["origin"])
            final_sample.SetSpacing(metadata["spacing"])
            sitk.WriteImage(final_sample, f"{target_directory}/{case_name}.nii.gz")


    def process(self) -> None:
        super().process()

    def get_folds(self, k: int) -> Dict[int, Dict[str, list]]:
        """
        Gets random fold at 80/20 split. Returns in a map.
        :param k: How many folds for kfold cross validation.
        :return: Folds map
        """
        splitter = Splitter(self.datapoints, k)
        return splitter.get_split_map()

    def process_fold(self, fold: int) -> None:
        """
        Preprocesses a fold. This method indirectly triggers saving of metadata if necessary,
        writes data to proper folder, and will perform any other future preprocessing.
        :param fold: The fold that we are currently preprocessing.
        :return: Nothing.
        """
        super().process_fold(fold)
