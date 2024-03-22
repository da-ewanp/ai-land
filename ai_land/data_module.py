import pytorch_lightning as pl
import torch
import xarray as xr
import yaml
from torch import tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split


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
        # path="normalise/ec_land_mean_std.zarr",
        path=CONFIG["normalise_path"],
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
    # load the dataset
    def __init__(
        self,
        start_yr=CONFIG["start_year"],
        end_yr=CONFIG["end_year"],
        x_idxs=CONFIG["x_slice_indices"],
        # x_idxs=(31294, 32294),
        path=CONFIG["file_path"],
    ):
        # List of climatological time-invariant features
        self.static_feat_lst = CONFIG["clim_feats"]
        # List of features that change in time
        self.dynamic_feat_lst = CONFIG["dynamic_feats"]  # + CONFIG["targets"]
        # Target list, make sure these are also the final features in feat_lst
        self.targ_lst = CONFIG["targets"]
        self.feat_lst = self.static_feat_lst + self.dynamic_feat_lst

        # Open datasets for times and x indices
        self.ds_ecland = (xr.open_zarr(path).sel(time=slice(start_yr, end_yr)).isel(x=slice(*x_idxs))).compute()
        self.ds_ecland_stat = (xr.open_zarr(path).sel(time=slice(start_yr, end_yr)).isel(x=slice(*x_idxs))).compute()

        self.x_size = self.ds_ecland.x.shape[0]
        self.time_size = self.ds_ecland.time.shape[0]

        # Define transforms for features and targets
        self.static_feat_scalar, self.dynamic_feat_scalar, self.targ_scalar = (
            TorchStandardScalerFeat(feat_lst=self.static_feat_lst),
            TorchStandardScalerFeat(feat_lst=self.dynamic_feat_lst),
            TorchStandardScalerFeat(feat_lst=self.targ_lst),
        )
        # Open datasets for the feature lists specified
        self.x_dynamic = self.ds_ecland[self.dynamic_feat_lst].to_array().astype("float32")
        if "time" in list(self.ds_ecland[self.static_feat_lst].to_array().dims):
            self.x_static = (
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
            self.x_static = (
                self.ds_ecland[self.static_feat_lst]
                .to_array()
                .astype("float32")
                .transpose(
                    "x",
                    "variable",
                )
                .values
            )
        self.x_static_scaled = tensor(
            self.static_feat_scalar.transform(self.X_static).reshape(1, self.x_size, -1),
            dtype=torch.float32,
        )

        self.y = self.ds_ecland[self.targ_lst].to_array().astype("float32")
        self.rollout = CONFIG["roll_out"]

    # number of rows in the dataset
    def __len__(self):
        return self.time_size - 1 - self.rollout

    # get a row at an index
    def __getitem__(self, idx):
        x = (
            self.x_dynamic.isel(time=slice(idx, idx + self.rollout + 1))
            .transpose(
                "time",
                "x",
                "variable",
            )
            .values
        )
        x = tensor(self.dynamic_feat_scalar.transform(x), dtype=torch.float32)
        y = (
            self.y.isel(time=slice(idx, idx + self.rollout + 1))
            .transpose(
                "time",
                "x",
                "variable",
            )
            .values
        )
        y = tensor(self.targ_scalar.transform(y), dtype=torch.float32)
        # Calculate delta_x update for corresponding x state
        y_inc = y[1:, :, :] - y[:-1, :, :]
        # Return self.x_static_scaled, X[:-1], Y[:-1], Y_inc
        return self.x_static_scaled.expand(self.rollout, -1, -1), x[:-1], y[:-1], y_inc

    # get indexes for train and test rows
    def get_splits(self, n_test=0.2):
        """_summary_

        :param n_test: _description_, defaults to 0.2 :return: _description_
        """
        # determine sizes
        generator = torch.Generator().manual_seed(42)
        test_size = round(n_test * (self.time_size - 1 - self.rollout))
        train_size = (self.time_size - 1 - self.rollout) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size], generator=generator)


class NonLinRegDataModule(pl.LightningDataModule):
    def setup(self, stage):
        torch.Generator().manual_seed(42)
        training_data = EcDataset()
        self.train, self.test = training_data.get_splits()

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=8)
