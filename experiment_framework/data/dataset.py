from torch.utils.data import Dataset
from torchvision import transforms 
from PIL import Image
import numpy as np

from hydra.utils import instantiate
from omegaconf import DictConfig

import pandas as pd
pd.options.mode.chained_assignment = None 


class ExprimentDataset(Dataset):
    def __init__(
        self, 
        df, 
        data_path,
        img_transform=None,
    ):
        self.df = df
        self.data_path = data_path
        self.img_transform = img_transform
        self.return_type = return_type

        self.setup()

    def __len__(self):
        return len(self.labels)

    def setup(self):
        """ 
        Setup the dataset by adding the image locations and labels
            self.images = []
            self.labels = []

        example
            self.df['image_locs'] = self.df.test_id.apply(
                lambda x: glob(f'{self.data_path}/{x}.webp')
            )
            self.df = self.df[self.df['image_locs'].map(lambda x: len(x)) > 0]
            self.df['image_locs'] = self.df['image_locs'].apply(lambda x: x[0])

            self.images = self.df['image_locs'].values
            self.labels = self.df['leftResult'].astype(np.float32).values
        """
        raise NotImplementedError


    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = np.array(self.labels[idx])

        # open image and get frame metadata
        image = Image.open(img_path)
        if self.img_transform:
            image = self.img_transform(image)

        return image, label
        

def get_datasets(cfg: DictConfig):
    """
    This function is a template for creating training and validation datasets.
    It takes a configuration object as input and returns a dictionary containing
    the training and validation datasets.

    The returned dictionary has the following structure:
    {
        'train': train_dataset, # Dataset object for training
        'valid': valid_dataset  # Dataset object for validation
    }

    Example usage:
    datasets = get_datasets(cfg)
    train_dataset = datasets['train']
    valid_dataset = datasets['valid']

    Note: This function currently raises a NotImplementedError as it is a template.
    The actual implementation will be added later.

    Args:
        cfg (DictConfig): The configuration object.

    Returns:
        dict: A dictionary containing the training and validation datasets.
    """
    raise NotImplementedError
