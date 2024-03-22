import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
import yaml
from torch import tensor
from torch.utils.data import DataLoader, Dataset, random_split


with open("config.yaml") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

torch.cuda.empty_cache()


# Define transform for features/targets
class TorchStandardScalerFeat:
    def __init__(
        self,
        feat_lst,
        path="normalise/ec_land_mean_std.zarr",
    ):
        self.ds_mustd = xr.open_zarr(path).sel(variable=feat_lst)
        self.mean = self.ds_mustd.var_mean.values
        self.std = self.ds_mustd.var_std.values

    def transform(self, x):
        x_norm = (x - self.mean) / (self.std + 1e-5)
        return x_norm

    def inv_transform(self, x_norm):
        x = (x_norm * (self.std + 1e-5)) + self.mean
        return x


# dataset definition
class EcDataset(Dataset):
    """Dataset class for ECLand zarr database
    """
    # load the dataset
    def __init__(
        self,
        start_yr=CONFIG["start_year"],
        end_yr=CONFIG["end_year"],
        x_idxs=CONFIG["x_slice_indices"],
        path=CONFIG["file_path"]
    ):
        # List of climatological time-invariant features
        self.static_feat_lst = CONFIG["clim_feats"]
        # List of features that change in time
        self.dynamic_feat_lst = CONFIG["dynamic_feats"] + CONFIG["targets"]
        # Target list, make sure these are also the final features in feat_lst
        self.targ_lst = CONFIG["targets"]
        self.feat_lst = self.static_feat_lst + self.dynamic_feat_lst

        # Open datasets for times and x indices
        self.ds_ecland = (
            xr.open_zarr(path).sel(time=slice(start_yr, end_yr)).isel(x=slice(*x_idxs))
        )  # .compute()
        self.ds_ecland_stat = (
            xr.open_zarr(path).sel(time=slice(start_yr, end_yr)).isel(x=slice(*x_idxs))
        )  # .compute()

        self.x_size = self.ds_ecland.x.shape[0]
        self.time_size = self.ds_ecland.time.shape[0]

        # Define transforms for features and targets
        self.static_feat_scalar, self.dynamic_feat_scalar, self.targ_scalar = (
            TorchStandardScalerFeat(feat_lst=self.static_feat_lst),
            TorchStandardScalerFeat(feat_lst=self.dynamic_feat_lst),
            TorchStandardScalerFeat(
                path="normalise/ec_land_deltax_mean_std.zarr", feat_lst=self.targ_lst
            ),
        )
        # Open datasets for the feature lists specified
        self.X_dynamic = (
            self.ds_ecland[self.dynamic_feat_lst].to_array().astype("float32")
        )
        if "time" in list(self.ds_ecland[self.static_feat_lst].to_array().dims):
            self.X_static = (
                self.ds_ecland[self.static_feat_lst]
                .to_array()
                .astype("float32")
                .transpose(
                    "time",
                    "x",
                    "variable",
                )
                .isel(time=0)
                .values
            )
        else:
            self.X_static = (
                self.ds_ecland[self.static_feat_lst]
                .to_array()
                .astype("float32")
                .transpose(
                    "x",
                    "variable",
                )
                .values
            )
        self.X_static_scaled = tensor(
            self.static_feat_scalar.transform(self.X_static).reshape(
                1, self.x_size, -1
            ),
            dtype=torch.float32,
        )
        # Set indexes and rollout
        self.targ_idx = np.array(
            [self.dynamic_feat_lst.index(var) for var in self.targ_lst]
        )
        # self.targ_idx_full = np.array(self.targ_idx) + self.X_static_scaled.shape[-1]
        self.rollout = CONFIG["roll_out"]

    # number of rows in the dataset
    def __len__(self):
        return self.time_size - 1 - self.rollout

    # get a row at an index
    def __getitem__(self, idx):
        X = (
            self.X_dynamic.isel(time=slice(idx, idx + self.rollout + 1))
            .transpose(
                "time",
                "x",
                "variable",
            )
            .values
        )
        X = self.dynamic_feat_scalar.transform(X)
        # Calculate delta_x update for corresponding x state
        Y = X[1:, :, self.targ_idx] - X[:-1, :, self.targ_idx]
        return [
            torch.cat(
                (
                    self.X_static_scaled.expand(self.rollout, -1, -1),
                    tensor(X[:-1, :, :], dtype=torch.float32),
                ),
                axis=-1,
            ),
            tensor(Y, dtype=torch.float32),
        ]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.2):
        # determine sizes
        generator = torch.Generator().manual_seed(42)
        test_size = round(n_test * (self.time_size - 1 - self.rollout))
        train_size = (self.time_size - 1 - self.rollout) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size], generator=generator)


class NonLinRegDataModule(pl.LightningDataModule):

    def setup(self, stage):
        generator = torch.Generator().manual_seed(42)
        training_data = EcDataset()
        self.train, self.test = training_data.get_splits()

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=8)
