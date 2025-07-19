from .darcy import DarcyDataset, load_darcy_2d
from .navier_stokes_incompressible import (
    NavierStokesIncompressibleDataset,
    load_ns_incom_2d,
)
from .navier_stokes_compressible import NavierStokesCompressibleDataset, load_ns_com
from .advection import AdvectionDataset, load_advection_1d
from .pt_dataset import PTDataset
from .burgers import BurgersDataset, load_burgers_1d
from .dict_dataset import DictDataset
from .mesh_datamodule import MeshDataModule
from .car_cfd_dataset import CarCFDDataset

# only import SphericalSWEDataset if torch_harmonics is built locally
try:
    from .spherical_swe import load_spherical_swe
except ModuleNotFoundError:
    pass
