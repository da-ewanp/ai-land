# Class for scaling features/targets
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
import zarr
from sklearn.metrics import r2_score
from torch import tensor

# Define the config for the experiment
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
    def __init__(
        self,
        input_size_clim,
        input_size_met,
        input_size_state,
        hidden_size,
        output_size,
        diag_output_size,
    ):
        super().__init__()
        # Normalization vector for delta_x's
        ds = zarr.open(CONFIG["file_path"])
        fistdiff_idx = [list(ds["variable"]).index(x) for x in CONFIG["targets_prog"]]
        self.ds_mean = tensor(ds.norm_firstdiff_means[fistdiff_idx])
        self.ds_std = tensor(ds.norm_firstdiff_stdevs[fistdiff_idx])

        #         # Define layers
        #         self.diag_output_size = diag_output_size
        #         input_dim = input_size_clim + input_size_met + input_size_state
        #         self.fc1 = nn.Linear(input_dim, hidden_size)
        #         self.relu1 = nn.ReLU()
        #         self.fc2 = nn.Linear(hidden_size, hidden_size)
        #         self.relu2 = nn.LeakyReLU()
        #         self.fc3 = nn.Linear(hidden_size, hidden_size)
        #         self.dropout = nn.Dropout(0.2)
        #         self.relu3 = nn.LeakyReLU()
        #         self.fc4 = nn.Linear(hidden_size, output_size)
        #         self.fc5 = nn.Linear(hidden_size, diag_output_size)

        #     def forward(self, clim_feats, met_feats, state_feats):
        #         combined = torch.cat((clim_feats, met_feats, state_feats), dim=-1)
        #         x = self.relu1(self.fc1(combined))
        #         x = self.dropout(self.relu2(self.fc2(x)))
        #         x = self.relu3(self.fc3(x))
        #         x_prog = self.fc4(x)
        #         x_diag = self.fc5(x)
        #         return x_prog, x_diag

        self.diag_output_size = diag_output_size
        self.layer_clim = nn.Linear(input_size_clim, hidden_size)
        self.relu1 = nn.Tanh()  # nn.ReLU()
        self.layer_met = nn.Linear(input_size_met, hidden_size)
        self.relu2 = nn.Tanh()  # nn.ReLU()
        self.layer_state = nn.Linear(input_size_state, hidden_size)
        self.relu3 = nn.Tanh()  # nn.ReLU()

        self.combine_layer = nn.Linear(hidden_size * 3, hidden_size)
        self.lrelu1 = nn.Tanh()  # nn.LeakyReLU()

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.lrelu2 = nn.Tanh()  # nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lrelu3 = nn.Tanh()  # nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.lrelu5 = nn.Tanh()  # nn.LeakyReLU()
        self.fc5 = nn.Linear(hidden_size, diag_output_size)

    def forward(self, clim_feats, met_feats, state_feats):
        out_clim = self.relu1(self.layer_clim(clim_feats))
        out_met = self.relu2(self.layer_met(met_feats))
        out_state = self.relu3(self.layer_state(state_feats))

        combined = torch.cat((out_clim, out_met, out_state), dim=-1)
        combined_out = self.lrelu1(self.combine_layer(combined))

        out = self.lrelu2(self.fc1(combined_out))
        out = self.lrelu3(self.fc2(out))
        out = self.fc3(out)

        out_diag = self.lrelu5(self.fc4(combined_out))
        out_diag = self.fc5(out_diag)
        return out, out_diag

    def transform(self, x, mean, std):
        x_norm = (x - mean) / (std + 1e-5)
        return x_norm

    def predict_step(
        self, clim_feats, met_feats, states, diagnostics
    ) -> Tuple[tensor, tensor]:
        """Given arrays of features produces a prediction for all timesteps.

        :return: (prognost_targets, diagnostic_targets)
        """
        preds = states.clone().to(self.device)
        preds_diag = diagnostics.clone().to(self.device)
        len_run = preds.shape[0]

        for x in range(len_run):
            preds_dx, preds_diag_x = self.forward(
                clim_feats, met_feats[[x]], preds[[x]]
            )
            if x < (len_run - 1):
                preds[x + 1] = preds[x] + preds_dx
            preds_diag[x] = preds_diag_x
        return preds, preds_diag

    def MSE_loss(self, logits, labels):
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()
        return criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x_clim, x_met, x_state, y, y_diag = train_batch
        logits, logits_diag = self.forward(x_clim, x_met, x_state)
        mean = self.ds_mean.to(self.device)
        std = self.ds_std.to(self.device)
        # loss = self.MSE_loss(logits / std, y / std)
        loss = self.MSE_loss(
            self.transform(logits, mean, std), self.transform(y, mean, std)
        )
        loss_abs = self.MSE_loss(x_state + logits, logits + y)
        loss_diag = self.MSE_loss(logits_diag, y_diag)
        self.log(
            "train_loss",
            loss,
        )  # on_step=False, on_epoch=True)
        self.log(
            "train_diag_loss",
            loss_diag,
        )

        if CONFIG["roll_out"] > 1:
            x_state_rollout = x_state.clone()
            y_rollout = y.clone()
            y_rollout_diag = y_diag.clone()
            for step in range(CONFIG["roll_out"]):
                # x = [batch_size=8, lookback (7) + rollout (3) = 10, n_feature = 37]
                x0 = x_state_rollout[
                    :, step, :, :
                ].clone()  # select input with lookback.
                y_hat, y_hat_diag = self.forward(
                    x_clim[:, step, :, :], x_met[:, step, :, :], x0
                )  # prediction at rollout step
                y_rollout_diag[:, step, :, :] = y_hat_diag
                if step < CONFIG["roll_out"] - 1:
                    x_state_rollout[:, step + 1, :, :] = (
                        x_state_rollout[:, step, :, :].clone() + y_hat
                    )  # overwrite x with prediction.
                y_rollout[:, step, :, :] = y_hat  # overwrite y with prediction.
            # step_loss = self.MSE_loss(y_rollout / std, y / std)
            step_loss = self.MSE_loss(
                self.transform(y_rollout, mean, std), self.transform(y, mean, std)
            )
            step_abs_loss = self.MSE_loss(x_state_rollout, x_state)
            step_loss_diag = self.MSE_loss(y_rollout_diag, y_diag)
            # step_loss = step_loss / ROLLOUT
            self.log(
                "step_loss",
                step_loss,
            )  # on_step=False, on_epoch=True)
            self.log(
                "step_loss_diag",
                step_loss_diag,
            )  # on_step=False, on_epoch=True)
            loss += step_loss
            loss_abs += step_abs_loss
            loss_diag += step_loss_diag

        return loss + loss_diag + step_abs_loss

    def validation_step(self, val_batch, batch_idx):
        x_clim, x_met, x_state, y, y_diag = val_batch
        mean = self.ds_mean.to(self.device)
        std = self.ds_std.to(self.device)
        logits, logits_diag = self.forward(x_clim, x_met, x_state)
        loss = self.MSE_loss(
            self.transform(logits, mean, std), self.transform(y, mean, std)
        )
        loss_diag = self.MSE_loss(logits_diag, y_diag)
        r2 = r2_score_multi(
            self.transform(logits, mean, std).cpu(),
            self.transform(y, mean, std).cpu(),
        )
        r2_diag = r2_score_multi(
            logits_diag.cpu(),
            y_diag.cpu(),
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_R2", r2, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "val_diag_loss", loss_diag, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log("val_diag_R2", r2_diag, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
