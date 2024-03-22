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
    """Calculated the r-squared score between 2 arrays of values.

    :param y_pred: predicted array :param y_true: "truth" array :return: r-squared
    metric
    """
    return r2_score(y_pred.flatten(), y_true.flatten())


# Define a neural network model with hidden layers and activation functions
class NonLinearRegression(pl.LightningModule):
    def __init__(self, input_size_clim, input_size_met, input_size_state, hidden_size, output_size):
        super().__init__()
        self.layer_clim = nn.Linear(input_size_clim, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer_met = nn.Linear(input_size_met, hidden_size)
        self.relu2 = nn.ReLU()
        self.layer_state = nn.Linear(input_size_state, hidden_size)
        self.relu3 = nn.ReLU()

        self.combine_layer = nn.Linear(hidden_size * 3, hidden_size * 2)
        self.lrelu1 = nn.LeakyReLU()

        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.lrelu2 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.lrelu3 = nn.LeakyReLU()
        self.lrelu3 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Target list
        self.targ_lst = CONFIG["targets"]
        path = CONFIG["normalise_delta_path"]
        self.ds_mustd = xr.open_zarr(path).sel(variable=self.targ_lst)
        self.mean = tensor(
            self.ds_mustd.var_mean.values,
            dtype=torch.float32,
        )
        self.std = tensor(
            self.ds_mustd.var_std.values,
            dtype=torch.float32,
        )

    def transform(self, x):
        x_norm = (x - self.mean.to(self.device)) / (self.std.to(self.device) + 1e-5)
        return x_norm

    def inv_transform(self, x_norm):
        x = (x_norm * (self.std.to(self.device) + 1e-5)) + self.mean.to(self.device)
        return x

    def forward(self, clim_feats, met_feats, state_feats):
        out_clim = self.relu1(self.layer_clim(clim_feats))
        out_met = self.relu2(self.layer_met(met_feats))
        out_state = self.relu3(self.layer_state(state_feats))

        combined = torch.cat((out_clim, out_met, out_state), dim=-1)
        combined_out = self.lrelu1(self.combine_layer(combined))

        out = self.lrelu2(self.fc1(combined_out))
        out = self.lrelu3(self.fc2(out))
        out = self.fc3(out)
        return out

    def predict(self, clim_feats, met_feats, state_feats):
        preds = state_feats.clone().to(self.device)
        for x in range(preds.shape[0] - 1):
            preds_dx = self.forward(clim_feats.to(self.device), met_feats[x].to(self.device), preds[x])
            preds[x + 1] = preds[x] + preds_dx
        return preds

    def MSE_loss(self, logits, labels):
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()
        return criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x_clim, x_met, x_state, y = train_batch
        logits = self.forward(x_clim, x_met, x_state)
        loss = self.MSE_loss(self.transform(logits), self.transform(y))
        self.log(
            "train_loss",
            loss,
        )  # on_step=False, on_epoch=True)

        if CONFIG["roll_out"] > 1:
            x_state_rollout = x_state.clone()
            y_rollout = y.clone()
            for step in range(CONFIG["roll_out"]):
                # x = [batch_size, rollout, x_dim, n_feature]
                y_hat = self.forward(
                    x_clim[:, step, :, :], x_met[:, step, :, :], x_state_rollout[:, step, :, :]
                )  # prediction at rollout step
                if step < CONFIG["roll_out"] - 1:
                    x_state_rollout[:, step + 1, :, :] = (
                        # x_state_rollout[:, step, :, :].clone() + y_hat
                        x_state_rollout[:, step, :, :]
                        + y_hat
                    )  # overwrite x with prediction.
                y_rollout[:, step, :, :] = y_hat  # overwrite y with prediction.
            step_loss = self.MSE_loss(self.transform(y_rollout), self.transform(y))
            # step_loss = step_loss / ROLLOUT
            self.log(
                "step_loss",
                step_loss,
            )  # on_step=False, on_epoch=True)
            loss += step_loss
        return loss

    def validation_step(self, val_batch, batch_idx):
        x_clim, x_met, x_state, y = val_batch
        logits = self.forward(x_clim, x_met, x_state)
        loss = self.MSE_loss(self.transform(logits), self.transform(y))
        r2 = r2_score_multi(
            self.transform(logits).cpu(),
            self.transform(y).cpu(),
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_r**2", r2, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
