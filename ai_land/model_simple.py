# Class for scaling features/targets
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import xarray as xr
import yaml
from sklearn.metrics import r2_score
from torch import tensor


with open("config.yaml") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

torch.cuda.empty_cache()


def r2_score_multi(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculated the r-squared score between 2 arrays of values

    :param y_pred: predicted array
    :param y_true: "truth" array
    :return: r-squared metric
    """
    return r2_score(y_pred.flatten(), y_true.flatten())


class TorchStandardScalerFeatTens:
    def __init__(self, feat_lst, path="normalise/ec_land_mean_std.zarr", dev="cpu"):
        self.ds_mustd = xr.open_zarr(path).sel(variable=feat_lst)
        self.mean = tensor(
            self.ds_mustd.var_mean.values, dtype=torch.float32, device=dev
        )
        self.std = tensor(self.ds_mustd.var_std.values, dtype=torch.float32, device=dev)

    def transform(self, x):
        x_norm = (x - self.mean) / (self.std + 1e-5)
        return x_norm

    def inv_transform(self, x_norm):
        x = (x_norm * (self.std + 1e-5)) + self.mean
        return x


# Define a neural network model with hidden layers and activation functions
class NonLinearRegression(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim, device):
        super(NonLinearRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        # List of climatological time-invariant features
        self.static_feat_lst = CONFIG["clim_feats"]
        # List of features that change in time
        self.dynamic_feat_lst = CONFIG["dynamic_feats"] + CONFIG["targets"]
        # Target list, make sure these are also the final features in feat_lst
        self.targ_lst = CONFIG["targets"]
        self.feat_lst = self.static_feat_lst + self.dynamic_feat_lst

        self.targ_scalar = TorchStandardScalerFeatTens(
            path="normalise/ec_land_deltax_mean_std.zarr",
            feat_lst=self.targ_lst,
            dev=device,
        )
        self.targ_idx = np.array(
            [self.dynamic_feat_lst.index(var) for var in self.targ_lst]
        )

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

    def MSE_loss(self, logits, labels):
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()
        return criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.MSE_loss(
            self.targ_scalar.transform(logits), self.targ_scalar.transform(y)
        )
        self.log(
            "train_loss",
            loss,
        )  # on_step=False, on_epoch=True)

        if CONFIG["roll_out"] > 1:
            x_rollout = x.clone()
            y_rollout = y.clone()
            for step in range(CONFIG["roll_out"]):
                # x = [batch_size=8, lookback (7) + rollout (3) = 10, n_feature = 37]
                x0 = x_rollout[:, step, :, :].clone()  # select input with lookback.
                y_hat = self.forward(x0)  # prediction at rollout step
                if step < CONFIG["roll_out"] - 1:
                    x_rollout[:, step + 1, :, self.targ_idx] = (
                        x_rollout[:, step, :, self.targ_idx].clone() + y_hat
                    )  # overwrite x with prediction.
                y_rollout[:, step, :, :] = y_hat  # overwrite y with prediction.
            step_loss = self.MSE_loss(
                self.targ_scalar.transform(y_rollout), self.targ_scalar.transform(y)
            )
            # step_loss = step_loss / ROLLOUT
            self.log(
                "step_loss",
                step_loss,
            )  # on_step=False, on_epoch=True)
            loss += step_loss

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.MSE_loss(
            self.targ_scalar.transform(logits), self.targ_scalar.transform(y)
        )
        r2 = r2_score_multi(
            self.targ_scalar.transform(logits).cpu(),
            self.targ_scalar.transform(y).cpu(),
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_r**2", r2, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
