import os
from typing import Tuple

import cftime
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
import zarr
from torch import tensor
from torch.utils.data import DataLoader, Dataset

# Open up experiment config
PATH_NAME = os.path.dirname(os.path.abspath(__file__))
with open(f"{PATH_NAME}/config.yaml") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

torch.cuda.empty_cache()


class Normalizer:
    def __init__(
        self,
        path=CONFIG["file_path"],
        normalize_dict=CONFIG["normalize"],
    ) -> None:
        self.ds_ecland = zarr.open(path)
        self.clim_stats = {
            "var_names": list(self.ds_ecland.clim_variable),
            "means": tensor(self.ds_ecland.clim_means),
            "stdevs": tensor(self.ds_ecland.clim_stdevs),
            "maxs": tensor(self.ds_ecland.clim_maxs),
            "mins": tensor(self.ds_ecland.clim_mins),
        }
        self.data_stats = {
            "var_names": list(self.ds_ecland.variable),
            "means": tensor(self.ds_ecland.data_means),
            "stdevs": tensor(self.ds_ecland.data_stdevs),
            "maxs": tensor(self.ds_ecland.data_maxs),
            "mins": tensor(self.ds_ecland.data_mins),
        }
        self.normalize_dict = normalize_dict

        self.clim_means, self.clim_stdevs = self.generate_norm_stats(
            CONFIG["clim_feats"], self.clim_stats
        )
        self.x_dynamic_means, self.x_dynamic_stdevs = self.generate_norm_stats(
            CONFIG["dynamic_feats"], self.data_stats
        )
        self.y_prog_means, self.y_prog_stdevs = self.generate_norm_stats(
            CONFIG["targets_prog"], self.data_stats
        )
        self.y_diag_means, self.y_diag_stdevs = self.generate_norm_stats(
            CONFIG["targets_diag"], self.data_stats
        )

    def transform(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Transform data with mean and stdev.

        :param x: data :param mean: mean :param std: standard deviation :return:
        normalised data
        """
        # x_norm = (x - mean) / (std + 1e-5)
        x_norm = (x - mean) / std
        return x_norm

    def inv_transform(
        self, x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        """Inverse transform on data with mean and stdev.

        :param x_norm: normlised data :param mean: mean :param std: standard deviation
        :return: unnormalised data
        """
        # x = (x_norm * (std + 1e-5)) + mean
        x = (x_norm * (std)) + mean
        return x

    def generate_norm_stats(self, var_list, stat_dict):
        means = torch.zeros(len(var_list))
        stdevs = torch.ones(len(var_list))
        for i, var in enumerate(var_list):
            var_idx = stat_dict["var_names"].index(var)
            if var in self.normalize_dict["max"]:
                stdevs[i] = stat_dict["maxs"][var_idx]
                # print(f"setting {var} normalisation=max, mean={means[i]}, stdev={stdevs[i]}")
            elif var in self.normalize_dict["none"]:
                means[i] = 0
                stdevs[i] = 1
            elif var in self.normalize_dict["stdev"]:
                stdevs[i] = stat_dict["stdevs"][var_idx]
                means[i] = 0
            elif var in self.normalize_dict["min-max"]:
                means[i] = stat_dict["mins"][var_idx]
                stdevs[i] = stat_dict["maxs"][var_idx] - stat_dict["mins"][var_idx]
                # print(f"setting {var} normalisation=none, mean={means[i]}, stdev={stdevs[i]}")
            # elif var in self.normalize_dict["minmax"]:
            #     means[i] = stat_dict["mins"][var_idx]
            #     stdevs[i] = stat_dict["maxs"][var_idx] - stat_dict["mins"][var_idx]
            #     print(f"setting {var} normalisation=minmax, mean={means[i]}, stdev={stdevs[i]}")
            # elif var in self.normalize_dict["stdev"]:
            #     stdevs[i] = stat_dict["stdevs"][var_idx]
            #     print(f"setting {var} normalisation=mean-std, mean={means[i]}, stdev={stdevs[i]}")
            else:
                means[i] = stat_dict["means"][var_idx]
                stdevs[i] = stat_dict["stdevs"][var_idx]
                # print(f"setting {var} normalisation=mean-std, mean={means[i]}, stdev={stdevs[i]}")
        return means, stdevs


class EcDataset(Dataset):
    # load the dataset
    def __init__(
        self,
        start_yr=CONFIG["start_year"],
        end_yr=CONFIG["end_year"],
        x_idxs=CONFIG["x_slice_indices"],
        path=CONFIG["file_path"],
        roll_out=CONFIG["roll_out"],
        # normalize_dict=CONFIG["normalize"],
        normalizer=None,
    ):
        self.path = path
        self.ds_ecland = zarr.open(path)
        # Create time index to select appropriate data range
        date_times = pd.to_datetime(
            cftime.num2pydate(
                self.ds_ecland["time"], self.ds_ecland["time"].attrs["units"]
            )
        )
        self.start_index = min(np.argwhere(date_times.year == int(start_yr)))[0]
        self.end_index = max(np.argwhere(date_times.year == int(end_yr)))[0]
        self.times = np.array(date_times[self.start_index : self.end_index])
        self.len_dataset = self.end_index - self.start_index

        # Select points in space
        self.x_idxs = (0, None) if "None" in x_idxs else x_idxs
        self.x_size = len(self.ds_ecland["x"][slice(*self.x_idxs)])
        self.lats = self.ds_ecland["lat"][slice(*self.x_idxs)]
        self.lons = self.ds_ecland["lon"][slice(*self.x_idxs)]

        self.normalizer = normalizer or Normalizer(path=self.path)

        # List of climatological time-invariant features
        self.static_feat_lst = CONFIG["clim_feats"]
        self.clim_index = [
            list(self.ds_ecland["clim_variable"]).index(x) for x in CONFIG["clim_feats"]
        ]
        # self.clim_means, self.clim_stdevs = self.generate_norm_stats(self.static_feat_lst, self.clim_stats)
        # List of features that change in time
        self.dynamic_feat_lst = CONFIG["dynamic_feats"]
        self.dynamic_index = [
            list(self.ds_ecland["variable"]).index(x) for x in CONFIG["dynamic_feats"]
        ]
        # self.x_dynamic_means, self.x_dynamic_stdevs = self.generate_norm_stats(self.dynamic_feat_lst, self.data_stats)
        # Prognostic target list
        self.targ_lst = CONFIG["targets_prog"]
        self.targ_index = [
            list(self.ds_ecland["variable"]).index(x) for x in CONFIG["targets_prog"]
        ]
        # self.y_prog_means, self.y_prog_stdevs = self.generate_norm_stats(self.targ_lst, self.data_stats)
        # Diagnostic target list
        self.targ_diag_lst = CONFIG["targets_diag"]
        self.targ_diag_index = [
            list(self.ds_ecland["variable"]).index(x) for x in CONFIG["targets_diag"]
        ]
        # self.y_diag_means, self.y_diag_stdevs = self.generate_norm_stats(self.targ_diag_lst, self.data_stats)

        # Create time-invariant static climatological features
        x_static = tensor(
            self.ds_ecland.clim_data[slice(*self.x_idxs), self.clim_index]
        )
        self.x_static_scaled = self.normalizer.transform(
            x_static, self.normalizer.clim_means, self.normalizer.clim_stdevs
        ).reshape(1, self.x_size, -1)

        # Define the statistics used for normalising the data
        self.y_prog_stdevs_test = tensor(self.ds_ecland.data_stdevs[self.targ_index])

        self.rollout = roll_out

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data into memory. **CAUTION ONLY USE WHEN WORKING WITH DATASET THAT FITS
        IN MEM**

        :return: static_features, dynamic_features, prognostic_targets,
        diagnostic_targets
        """
        ds_slice = tensor(
            self.ds_ecland.data[
                self.start_index : self.end_index, slice(*self.x_idxs), :
            ]
        )

        X = ds_slice[:, :, self.dynamic_index]
        X = self.normalizer.transform(
            X, self.normalizer.x_dynamic_means, self.normalizer.x_dynamic_stdevs
        )

        X_static = self.x_static_scaled

        Y_prog = ds_slice[:, :, self.targ_index]
        Y_prog = self.normalizer.transform(
            Y_prog, self.normalizer.y_prog_means, self.normalizer.y_prog_stdevs
        )

        Y_diag = ds_slice[:, :, self.targ_diag_index]
        Y_diag = self.normalizer.transform(
            Y_diag, self.normalizer.y_diag_means, self.normalizer.y_diag_stdevs
        )
        return X_static, X, Y_prog, Y_diag

    # number of rows in the dataset
    def __len__(self):
        return self.len_dataset - 1 - self.rollout

    # get a row at an index
    def __getitem__(self, idx):
        idx = idx + self.start_index

        ds_slice = tensor(
            self.ds_ecland.data[
                slice(idx, idx + self.rollout + 1), slice(*self.x_idxs), :
            ]
        )

        X_static = self.x_static_scaled.expand(self.rollout, -1, -1)

        X = ds_slice[:, :, self.dynamic_index]
        X = self.normalizer.transform(
            X, self.normalizer.x_dynamic_means, self.normalizer.x_dynamic_stdevs
        )

        Y_prog = ds_slice[:, :, self.targ_index]
        Y_prog = self.normalizer.transform(
            Y_prog, self.normalizer.y_prog_means, self.normalizer.y_prog_stdevs
        )

        Y_diag = ds_slice[:, :, self.targ_diag_index]
        Y_diag = self.normalizer.transform(
            Y_diag, self.normalizer.y_diag_means, self.normalizer.y_diag_stdevs
        )

        return X_static, X[:-1], Y_prog[:-1], Y_diag[:-1]


class NonLinRegDataModule(pl.LightningDataModule):
    """Pytorch lightning specific data class."""

    def setup(self, stage):
        # generator = torch.Generator().manual_seed(42)
        self.train = EcDataset(start_yr=CONFIG["start_year"], end_yr=CONFIG["end_year"])
        self.test = EcDataset(
            start_yr=CONFIG["validation_start"], end_yr=CONFIG["validation_end"]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
            persistent_workers=True,
            pin_memory=True,
        )
