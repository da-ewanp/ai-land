from setuptools import find_packages
from setuptools import setup

setup(
    name="ai-land",
    version="0.0.1",
    author="Ewan Pinnington",
    author_email="ewan.pinnington@ecmwf.int",
    description="Experimental land surface model emulator",
    packages=find_packages(),
    install_requires=[
        "pyyaml",
        "black",
        "isort",
        "flake8",
        "pytest",
        "ipython",
        "cartopy",
        "bokeh==2.4.3",
        "xarray==2023.1.0",
        "torch>=2.2",
        "pytorch-lightning>=2.1.0",
        "matplotlib>=3.7.1",
        "torchinfo>=1.8.0",
        "wandb>=0.15.0",
        "zarr>=2.14.2",
        "pre-commit>=3.3.3",
        "torch-cluster",
        "earthkit",
    ],
)
