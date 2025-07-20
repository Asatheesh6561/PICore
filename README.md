# PICore
Source code for the paper "PICore: Physics-Informed Unsupervised Coreset Selection for Data Efficient Neural Operator Training".

## Installation
```
git clone https://github.com/Asatheesh6561/PICore && cd PICore
conda env create -f environment.yml
conda activate picore
```

## Data Generation
For the Advection, Burgers, and Darcy datasets, we use the data generation scripts found in PDEBench. For the Navier Stokes Incompressible dataset, we provide generation code in ```data_generation/ns_incompressible.py```. To downsample to smaller resolutions for training, use the scripts in ```load_data```. We have also provided pre-generated data in this
[url](https://drive.google.com/drive/folders/1aypSBwUhdjH5_HxlcxfL1V4p1BYdu9yc?usp=sharing).

## Training
We use hydra for configuration files for managing data-specific and model-specific parameters. For example, running PICore on the Advection Dataset with FNO using CRAIG as the coreset selection algorithm, run
```
python train.py model=FNO dataset=Advection coreset_algorithm=craig
```

## Logging
We provide optional logging to Weights and Biases, but you must change the wandb parameters in ```configs/config.yaml``` if you are using it. We also save all results to a pickle file in the ```results``` folder.


