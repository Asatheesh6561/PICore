# PICore

# Installation
```
git clone https://github.com/Asatheesh6561/PICore
conda env create -f environment.yml
```

# Data Generation
For the Advection, Burgers, and Darcy datasets, we use the data generation scripts found in PDEBench. For the Navier Stokes Incompressible dataset, we provide generation code in ```data_generation/ns_incompressible.py```. To downsample to smaller resolutions for training, use the scripts in ```load_data```.

# Usage
We use a modified version of the (neuraloperator)[https://github.com/Asatheesh6561/neuraloperator] library for loading PDE datasets and Neural Operator Implementations.
