from pathlib import Path
from typing import Optional, List, Union

import numpy as np
import torch

from torch.utils.data import DataLoader

from .pt_dataset import PTDataset


class BurgersDataset(PTDataset):
    """
    BurgersDataset wraps data from the burgers equation in 1 spatial dimension.

    Parameters
    ----------
    root_dir : Union[Path, str]
        root at which to download data files
    n_train : int
        number of train instances
    n_tests : List[int]
        number of test instances per test dataset
    batch_size : int
        batch size of training set
    test_batch_sizes : List[int]
        batch size of test sets
    train_resolution : int
        resolution of data for training set
    test_resolutions : List[int], optional
        resolution of data for testing sets, by default [16]
    temporal_subsample : int, optional
        rate at which to subsample the temporal dimension, by default None
    spatial_subsample : int, optional
        rate at which to subsample along the spatial dimension, by default None

    Attributes
    ----------
    train_db: torch.utils.data.Dataset of training examples
    test_db:  ""                       of test examples
    data_processor: neuralop.data.transforms.DataProcessor to process data examples
        optional, default is None
    """

    def __init__(
        self,
        root_dir: Union[Path, str],
        n_train: int,
        n_tests: list[int],
        train_resolution: int = 16,
        test_resolutions: List[int] = [16],
        batch_size: int = 32,
        test_batch_sizes: List[int] = 32,
        temporal_subsample: Optional[int] = None,
        spatial_subsample: Optional[int] = None,
        pad: int = 0,
    ):
        # convert root dir to path
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        super().__init__(
            root_dir=root_dir,
            n_train=n_train,
            n_tests=n_tests,
            batch_size=batch_size,
            test_batch_sizes=test_batch_sizes,
            train_resolution=train_resolution,
            test_resolutions=test_resolutions,
            input_subsampling_rate=spatial_subsample,
            output_subsampling_rate=spatial_subsample,
            encode_input=True,
            encode_output=True,
            encoding="channel-wise",
            channel_dim=1,
            channels_squeezed=True,
            dataset_name="burgers",
        )


def load_burgers_1d(
    data_root: Union[Path, str],
    n_train: int,
    n_test: int,
    batch_size: int,
    test_batch_sizes: List[int],
    train_resolution: int = 16,
    test_resolutions: List[int] = [16],
    temporal_subsample: Optional[int] = 1,
    spatial_subsample: Optional[int] = 1,
):
    """
    Legacy function to load the Burgers equation dataset

    Parameters
    ----------
    data_root : Union[Path, str]
        Path to the directory containing data files
    n_train : int
        Number of training instances
    n_test : int
        Number of testing instances
    batch_size : int
        Batch size for the training set
    test_batch_sizes : List[int]
        Batch sizes for the test sets
    train_resolution : int, optional
        Resolution of the training data, default is 16
    test_resolutions : List[int], optional
        Resolutions for the test data, default is [16]
    temporal_subsample : int, optional
        Rate at which to subsample the temporal dimension, default is 1
    spatial_subsample : int, optional
        Rate at which to subsample the spatial dimension, default is 1
    """
    burgers_dataset = BurgersDataset(
        root_dir=data_root,
        n_train=n_train,
        n_tests=n_test,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        train_resolution=train_resolution,
        test_resolutions=test_resolutions,
        temporal_subsample=temporal_subsample,
        spatial_subsample=spatial_subsample,
    )
    train_loader = DataLoader(
        burgers_dataset.train_db, batch_size=batch_size, shuffle=True
    )

    test_loaders = {
        resolution: DataLoader(
            burgers_dataset.test_dbs[resolution],
            batch_size=test_batch_sizes[i],
            shuffle=False,
        )
        for i, resolution in enumerate(test_resolutions)
    }

    return train_loader, test_loaders, burgers_dataset.data_processor
