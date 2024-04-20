# ECMWF Land Surface Emulator AI-LAND

This repo contains logic to train a Neural Network using Pytorch to emulate the ECMWF land surface model ECLand. The training is configured within the `ai_land/config.yaml` where the features/targets, file paths, batch size, *etc*. can be specified.

The dataset is loaded in `ai_land/data_module.py`, with the model being defined in `ai_land/model.py` and the training controlled by `ai_land/training.py`. There are also some examples of training callbacks, plotting intermediate results during training, in `ai_land/train_callbacks.py`.

Under `notebooks/` we include some examples of running the model and comparing the output with that of the full ECLand model for a year that is independent from the training period.

## Quick Start

The necessary Python dependencies are included in the `setup.py` file. The project can be installed under a conda or virtual environment. After activating your new environment and navigating to the `ai-land` directory simply run:
```
pip install -e .
```
If you are contributing to this repo we also are using pre-commit hooks to keep the code readable. Please install these using:
```
pre-commit install
```
Then the code will automatically be reformatted when you commit any changes via Git. Please also ensure you raise a PR for any changes going to a main branch and get these peer-reviewed. Thank you! :pray:

![ec/ai-land comparison](docs/ai-land-comp.gif "ai-land")
