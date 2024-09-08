# Class for scaling features/targets
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from sklearn.metrics import r2_score
from torch import tensor

# Define the config for the experiment
PATH_NAME = os.path.dirname(os.path.abspath(__file__))
with open(f"{PATH_NAME}/config.yaml") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

torch.cuda.empty_cache()


def make_map_val_plot(pred_arr, targ_arr, lat_arr, lon_arr, name_lst):
    fig, axes = plt.subplots(
        nrows=3,
        ncols=int(np.ceil(len(name_lst) / 3)),
        figsize=(18, 9),
    )

    map_errs = 100 * np.abs((pred_arr - targ_arr) / (targ_arr + 1e-5))
    # map_errs = 100 * np.abs((pred_arr - targ_arr) / np.mean(targ_arr))
    mape = np.mean(map_errs, axis=(0, 1))

    for i, axis in enumerate(axes.flatten()):
        if i < len(name_lst):
            var = name_lst[i]
            c = axis.scatter(
                lon_arr[::1],
                lat_arr[::1],
                c=mape[::1, name_lst.index(var)],
                vmin=0,
                vmax=100,
                s=1,
            )
            plt.colorbar(c)
            axis.set_title(f"MAPE {var}")
        else:
            axis.set_axis_off()

    fig.tight_layout()
    return fig


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
        mu_norm=0,
        std_norm=1,
        dataset=None,
    ):
        super().__init__()
        # Normalization vector for delta_x's
        self.mu_norm = tensor(mu_norm)
        self.std_norm = tensor(std_norm)
        self.ds = dataset

        # Define layers
        self.diag_output_size = diag_output_size
        input_dim = input_size_clim + input_size_met + input_size_state

        num_blocks = 2  # 4  # 5 # 6
        hidden_dim = hidden_size
        output_dim = output_size

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.skip = nn.Linear(output_dim, output_dim)
        
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                # nn.LeakyReLU(),
                # nn.GELU(),
                nn.LayerNorm(hidden_dim),
                
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.1)
            
            ) for _ in range(num_blocks)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_proj_diag = nn.Linear(hidden_dim, diag_output_size)

    def forward(self, clim_feats, met_feats, state_feats):
        x = torch.cat((clim_feats, met_feats, state_feats), dim=-1)
        # Initial projection
        h = self.input_proj(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            h = h + block(h)  # Residual connection
            #h = block(h)
        
        # Final projection
        out = self.output_proj(h)
        out_diag = self.output_proj_diag(h)
        
        # Optional: Add the input to the output for a stronger residual effect
        out = out + self.skip(state_feats)  # Uncomment if input_dim == output_dim
        #out = out + state_feats  # Uncomment if input_dim == output_dim   
        return out, out_diag

    def transform(self, x, mean, std):
        x_norm = (x - mean) / (std + 1e-5)
        # x_norm = (x - mean) / (std)
        return x_norm

    def predict_step(
        self, clim_feats, met_feats, states, diagnostics
    ) -> Tuple[tensor, tensor]:
        """Given arrays of features produces a prediction for all timesteps.

        :return: (prognost_targets, diagnostic_targets)
        """
        preds = states.clone().to(self.device)
        preds_diag = diagnostics.clone().to(self.device)
        # preds = torch.zeros_like(states).to(self.device)
        # preds_diag = torch.zeros_like(diagnostics).to(self.device)
        # preds[0] = states[0]
        len_run = preds.shape[0]

        for x in range(len_run):
            preds_x, preds_diag_x = self.forward(
                clim_feats, met_feats[[x]], preds[[x]]
            )
            if x < (len_run - 1):
                preds[x + 1] = preds_x  # + preds_dx
            preds_diag[x] = preds_diag_x
        return preds, preds_diag

    def MSE_loss(self, logits, labels):
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()
        return criterion(logits, labels)
    
    def _step(self, batch, batch_idx):
        x_clim, x_met, x_state, y_diag = batch
        mean = self.mu_norm.to(self.device)
        std = self.std_norm.to(self.device)
        loss = torch.zeros(1, device=self.device, requires_grad=False)

        for rollout_step in range(CONFIG["roll_out"]):
            x0 = x_state[:, rollout_step, :, :].clone()
            y_hat, y_hat_diag = self.forward(
                x_clim[:, rollout_step, :, :], x_met[:, rollout_step, :, :], x0
            )
            y_hat -= x0

            if rollout_step < CONFIG["roll_out"] - 1:
                y = x_state[:, rollout_step + 1, :, :] - x0
                x_state[:, rollout_step + 1, :, :] = (
                    # x_state[:, rollout_step, :, :].clone() + y_hat
                    x0 + y_hat
                )  # overwrite x with prediction

                loss += self.MSE_loss(self.transform(y_hat, mean, std), self.transform(y, mean, std))
                loss += self.MSE_loss(y_hat_diag, y_diag[:, rollout_step, :, :])
        
        # scale loss
        loss *= 1.0 / CONFIG["roll_out"]
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch, batch_idx)
        self.log("loss", 
                 loss, 
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 batch_size=train_batch[0].shape[0],
                 sync_dist=True,)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            loss = self._step(val_batch, batch_idx)
        self.log("val_loss", loss,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 batch_size=val_batch[0].shape[0],
                 sync_dist=True,)
        return loss

        # if ((self.current_epoch + 1) % CONFIG["logging"]["plot_freq"] == 0) & (
        #     batch_idx == 0
        # ):
        #     self.log_fig_mlflow(x_state_rollout, x_state, mean, std)

    @rank_zero_only
    def log_fig_mlflow(self, logits, y, mean, std):
        fig = make_map_val_plot(
            # (x_state + logits).cpu().numpy(),
            # (x_state + y).cpu().numpy(),
            # self.transform(logits, mean, std).cpu().numpy(),
            # self.transform(y, mean, std).cpu().numpy(),
            logits.cpu().numpy(),
            y.cpu().numpy(),
            self.ds.lats,
            self.ds.lons,
            self.ds.targ_lst,
        )
        if CONFIG["logging"]["logger"] == "mlflow":
            self.logger.experiment.log_figure(
                self.logger.run_id,
                fig,
                f"map_val_epoch{self.current_epoch + 1}.png",
            )
        else:
            fig.savefig(
                f"{CONFIG['logging']['location']}/plots/map_val_epoch{self.current_epoch + 1}.png"
            )
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
