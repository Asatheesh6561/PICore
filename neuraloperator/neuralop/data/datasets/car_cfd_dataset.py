from typing import List, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .mesh_datamodule import MeshDataModule
from .web_utils import download_from_zenodo_record

class CarCFDDataset(MeshDataModule):
    """CarCFDDataset is a processed version of the dataset introduced in
    [1]_, which encodes a triangular mesh over the surface of a 3D model car
    and provides the air pressure at each centroid and vertex of the mesh when
    the car is placed in a simulated wind tunnel with a recorded inlet velocity.
    In our case, inputs are a signed distance function evaluated over a regular
    3D grid of query points, as well as the inlet velocity. Outputs are pressure 
    values at each centroid of the triangle mesh.
    
        .. warning:: 

        ``CarCFDDataset`` inherits from ``MeshDataModule``, which requires the optional ``open3d`` dependency.
        See :ref:`open3d_dependency` for more information. 

    We also add additional manifest files to split the provided examples
    into training and testing sets, as well as remove instances that are corrupted.

    Data is also stored on Zenodo: https://zenodo.org/records/13936501

    Parameters
    ----------
    root_dir : Union[str, Path]
        root directory at which data is stored.
    n_train : int, optional
        Number of training instances to load, by default 1
    n_test : int, optional
        Number of testing instances to load, by default 1
    query_res : List[int], optional
        Dimension-wise resolution of signed distance function 
        (SDF) query cube, by default [32,32,32]
    download : bool, optional
        Whether to download data from Zenodo, by default True
    

    Attributes
    ----------
    train_loader: torch.utils.data.DataLoader
        dataloader of training examples
    test_loader: torch.utils.data.DataLoader
        dataloader of testing examples

    References
    ----------
    .. [1] :
    
    Umetani, N. and Bickel, B. (2018). "Learning three-dimensional flow for interactive 
        aerodynamic design". ACM Transactions on Graphics, 2018. 
        https://dl.acm.org/doi/10.1145/3197517.3201325.
    """

    def __init__(self,
        root_dir: Union[str, Path],
        n_train: int = 1,
        n_test: int = 1,
        query_res: List[int] = [32,32,32],
        download: bool=True
        ):
        """Initialize the CarCFDDataset.
        """
        self.zenodo_record_id = "13936501"

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        if not root_dir.exists():
            root_dir.mkdir(parents=True)
        
        if download:
            download_from_zenodo_record(record_id=self.zenodo_record_id,
                                        root=root_dir)
        super().__init__(
            root_dir=root_dir,
            item_dir_name='',
            n_train=n_train,
            n_test=n_test,
            query_res=query_res,
            attributes=['press']
        )

        # process data list to remove specific vertices from pressure to match number of vertices
        for i, data in enumerate(self.train_data.data_list):
            press = data['press']
            self.train_data.data_list[i]['press'] = torch.cat((press[:,0:16], press[:,112:]), axis=1)
        for i, data in enumerate(self.test_data.data_list):
            press = data['press']
            self.test_data.data_list[i]['press'] = torch.cat((press[:,0:16], press[:,112:]), axis=1)


def load_car_cfd(
    data_root: Union[Path, str],
    n_train: int,
    n_test: int,
    batch_size: int,
    test_batch_size: int,
    query_res: List[int] = [32, 32, 32],
    download: bool = True,
):
    """
    Legacy function to load the CarCFD dataset

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
    test_batch_size : int
        Batch size for the test set
    query_res : List[int], optional
        Dimension-wise resolution of signed distance function (SDF) query cube, default is [32, 32, 32]
    download : bool, optional
        Whether to download data from Zenodo, default is True

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test set
    data_processor : None
        Data processor (None for CarCFDDataset as it inherits from MeshDataModule)
    """
    car_cfd_dataset = CarCFDDataset(
        root_dir=data_root,
        n_train=n_train,
        n_test=n_test,
        query_res=query_res,
        download=download,
    )

    train_loader = DataLoader(
        car_cfd_dataset.train_data, batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(
        car_cfd_dataset.test_data, batch_size=test_batch_size, shuffle=False
    )

    return train_loader, test_loader, None
