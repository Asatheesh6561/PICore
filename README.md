# PICore
This repository provides source code for the TMLR 2025 paper "[PICore: Physics-Informed Unsupervised Coreset Selection for Data Efficient Neural Operator Training](https://openreview.net/pdf?id=l0VSewTJCI)".

## Installation
```
git clone https://github.com/Asatheesh6561/PICore && cd PICore
conda env create -f environment.yml
conda activate picore
```

## Data Generation
For the Advection, Burgers, and Darcy datasets, we use the data generation scripts found in PDEBench. For the Navier Stokes Incompressible dataset, we provide generation code in ```data_generation/ns_incompressible.py```. To downsample to smaller resolutions for training, use the scripts in ```load_data```. We have also provided pre-generated data in Huggingface Repo: [https://huggingface.co/datasets/asatheesh/PICore](https://huggingface.co/datasets/asatheesh/PICore).

## Training
We use hydra for configuration files for managing data-specific and model-specific parameters. For example, running PICore on the Advection Dataset with FNO using CRAIG as the coreset selection algorithm, run
```
python main.py model=FNO dataset=Advection coreset_algorithm=craig
```

## Logging
We provide optional logging to Weights and Biases, but you must change the wandb parameters in ```configs/config.yaml``` if you are using it. We also save all results to a pickle file in the ```results``` folder.

## Citation
PICore is accepted to Transactions on Machine Learning Research 2025. If you use PICore in your work, please cite us as follows:
```bibtex
@article{
  satheesh2025picore,
  title={{PIC}ore: Physics-Informed Unsupervised Coreset Selection for Data Efficient Neural Operator Training},
  author={Anirudh Satheesh and Anant Khandelwal and Mucong Ding and Radu Balan},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=l0VSewTJCI},
  note={}
}
```


