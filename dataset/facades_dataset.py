import cv2
import os
import numpy as np

from torch.utils.data import Dataset
from typing import Callable


class FacadesDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split: str = 'test',
                 transformations: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        assert split in ['test', 'train', 'val'], \
            f"Unknown split {split}."
        self.__split = split

        self.__root_dir = root_dir
        self.__images = os.listdir(os.path.join(self.__root_dir, self.__split))
        self.__transformations = transformations

    def __len__(self) -> int:
        return len(self.__images)

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        image_filepath = os.path.join(self.__root_dir, self.__split, self.__images[index])
        image_atlas = cv2.imread(image_filepath)

        shape = image_atlas.shape
        width = shape[1]

        image = image_atlas[:, :(width//2)]
        labels = image_atlas[:, (width//2):]

        if self.__transformations is not None:
            image, labels = self.__transformations(image, labels)

        return image, labels
