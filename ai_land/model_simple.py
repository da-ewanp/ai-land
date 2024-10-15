# Class for scaling features/targets
import os
from typing import Tuple

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from plot import make_map_val_plot
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import tensor

# Define the config for the experiment
PATH_NAME = os.path.dirname(os.path.abspath(__file__))
with open(f"{PATH_NAME}/config.yaml") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

torch.cuda.empty_cache()


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
        lower_bounded_indices_prog=None,  # Indices of prognostic features that should be lower-bounded (>= 0)
        lower_bounded_indices_diag=None,  # Indices of diagnostic features that should be lower-bounded (<= 1)
        upper_bounded_indices_prog=None,  # Indices of prognostic features that should be upper-bounded (<= 1)
        upper_bounded_indices_diag=None,  # Indices of diagnostic features that should be upper-bounded (<= 1)
        upper_lower_bounded_indices_prog=None,  # Indices of prognostic features that should be upper-lower-bounded (>= 0, <= 1)
        upper_lower_bounded_indices_diag=None,  # Indices of diagnostic features that should be upper-lower-bounded (>= 0, <= 1)
    ):
        super().__init__()
        # Normalization vector for delta_x's
        self.mu_norm = tensor(mu_norm)
        self.std_norm = tensor(std_norm)
        # self.std_norm[-1] = self.std_norm[-1]*0.4
        # self.std_norm[-2] = self.std_norm[-2]*0.2
        self.ds = dataset

        self.lower_bounded_indices_prog = lower_bounded_indices_prog or []
        self.lower_bounded_indices_diag = lower_bounded_indices_diag or []
        self.upper_bounded_indices_prog = upper_bounded_indices_prog or []
        self.upper_bounded_indices_diag = upper_bounded_indices_diag or []
        self.upper_lower_bounded_indices_prog = upper_lower_bounded_indices_prog or []
        self.upper_lower_bounded_indices_diag = upper_lower_bounded_indices_diag or []

        # Define layers
        self.diag_output_size = diag_output_size
        input_dim = input_size_clim + input_size_met + input_size_state

        num_blocks = CONFIG["num_blocks"]  # 3  # 5 # 6  ** 4 worked!! **
        hidden_dim = hidden_size
        output_dim = output_size

        layers = []
        hidden_sizes = [hidden_dim] * num_blocks
        # hidden_sizes = [256, 128, 64, 32, 16]
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            # layers.append(nn.Tanh())
            prev_size = hidden_size
        # layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

        layers_diag = []
        prev_size = input_dim
        for hidden_size in [hidden_dim] * 1:
            layers_diag.append(nn.Linear(prev_size, hidden_size))
            layers_diag.append(nn.LayerNorm(hidden_size))
            layers_diag.append(nn.ReLU())
            # layers.append(nn.Tanh())
            prev_size = hidden_size
        # layers.append(nn.Linear(prev_size, output_size))
        self.model_diag = nn.Sequential(*layers_diag)

        # self.input_proj = nn.Linear(input_dim, hidden_dim)
        # self.skip = nn.Linear(output_dim, output_dim)

        # self.residual_blocks = nn.ModuleList([
        #     nn.Sequential(
        #         # nn.Linear(hidden_dim, hidden_dim),
        #         # nn.ReLU(),
        #         nn.LeakyReLU(),
        #         # nn.GELU(),
        #         # nn.LayerNorm(hidden_dim),
        #         nn.Linear(hidden_dim, hidden_dim),
        #         # nn.Dropout(0.1)
        #     ) for _ in range(num_blocks)
        # ])

        self.output_proj = nn.Linear(hidden_sizes[-1], output_dim)
        self.output_proj_diag = nn.Linear(hidden_sizes[-1], diag_output_size)

    def smooth_clamp(self, x, max_val=1.0, k=50):
        return max_val - torch.log(1 + torch.exp(k * (max_val - x))) / k

    def forward(self, clim_feats, met_feats, state_feats):
        x = torch.cat((clim_feats, met_feats, state_feats), dim=-1)
        # Initial projection
        # h = self.input_proj(x)

        # Residual blocks
        # for block in self.residual_blocks:
        # for block in self.model:
        #    h = h + block(h)  # Residual connection
        # h = block(h)
        x = self.model(x)

        # Final projection
        out = self.output_proj(x)
        # out_diag = self.output_proj_diag(x)

        # Optional: Add the input to the output for a stronger residual effect
        # out = out + self.skip(state_feats)  # Uncomment if input_dim == output_dim
        out = out + state_feats  # Uncomment if input_dim == output_dim

        out_diag = self.output_proj_diag(
            self.model_diag(torch.cat((clim_feats, met_feats, out), dim=-1))
        )

        # Apply bounds to specific feature indices of the final output
        if self.lower_bounded_indices_prog:
            # Create masks for the features to be bounded
            lower_mask_prog = torch.zeros_like(out, dtype=torch.bool)
            lower_mask_prog[..., self.lower_bounded_indices_prog] = True
            # Apply Relu to lower-bounded features
            out = torch.where(lower_mask_prog, torch.relu(out), out)
        if self.lower_bounded_indices_diag:
            # Create masks for the features to be bounded
            lower_mask_diag = torch.zeros_like(out_diag, dtype=torch.bool)
            lower_mask_diag[..., self.lower_bounded_indices_diag] = True
            # Apply Relu to lower-bounded features
            out_diag = torch.where(lower_mask_diag, torch.relu(out_diag), out_diag)
        if self.upper_bounded_indices_prog:
            # Create masks for the features to be bounded
            upper_mask_prog = torch.zeros_like(out, dtype=torch.bool)
            upper_mask_prog[..., self.upper_bounded_indices_prog] = True
            # Apply Relu to lower-bounded features
            out = torch.where(upper_mask_prog, out - torch.relu(out - 1), out)
        if self.upper_bounded_indices_diag:
            # Create masks for the features to be bounded
            upper_mask_diag = torch.zeros_like(out_diag, dtype=torch.bool)
            upper_mask_diag[..., self.upper_bounded_indices_diag] = True
            # Apply Relu to lower-bounded features
            out_diag = torch.where(
                upper_mask_diag, out_diag - torch.relu(out_diag - 1), out_diag
            )
        if self.upper_lower_bounded_indices_prog:
            # Create masks for the features to be bounded
            upper_lower_mask_prog = torch.zeros_like(out, dtype=torch.bool)
            upper_lower_mask_prog[..., self.upper_lower_bounded_indices_prog] = True
            # Apply Relu to lower-bounded features
            out = torch.where(
                upper_lower_mask_prog,
                torch.min(torch.relu(out), torch.tensor(1.0)),
                out,
            )
        if self.upper_lower_bounded_indices_diag:
            # Create masks for the features to be bounded
            upper_lower_mask_diag = torch.zeros_like(out_diag, dtype=torch.bool)
            upper_lower_mask_diag[..., self.upper_lower_bounded_indices_diag] = True
            # Apply Relu to lower-bounded features
            out_diag = torch.where(
                upper_lower_mask_diag,
                torch.min(torch.relu(out_diag), torch.tensor(1.0)),
                out_diag,
            )
        return out, out_diag

    def transform(self, x, mean, std):
        # x_norm = (x - mean) / (std + 1e-5)
        # x_norm = (x - mean) / (std)
        # x_norm = x / std
        x_norm = x / std
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
            preds_x, preds_diag_x = self.forward(clim_feats, met_feats[[x]], preds[[x]])
            if x < (len_run - 1):
                # print("********* !!!! ********* iter", x, len_run, x+1)
                preds[x + 1] = preds_x  # + preds_dx
            preds_diag[x] = preds_diag_x
        return preds, preds_diag

    def MSE_loss(self, logits, labels):
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()
        return criterion(logits, labels)

    def _step(self, batch, batch_idx, val: bool = False):
        x_clim, x_met, x_state, y_diag = batch
        mean = self.mu_norm.to(self.device)
        std = self.std_norm.to(self.device)
        loss = torch.zeros(1, device=self.device, requires_grad=False)

        for rollout_step in range(CONFIG["roll_out"] - 1):
            x0 = x_state[:, rollout_step, :, :].clone()
            y_hat, y_hat_diag = self.forward(
                x_clim[:, rollout_step, :, :], x_met[:, rollout_step, :, :], x0
            )
            y = x_state[:, rollout_step + 1, :, :].clone()  # - x0
            x_state[:, rollout_step + 1, :, :] = y_hat  # overwrite x with prediction
            y_diag_step = y_diag[:, rollout_step, :, :].clone()

            loss += self.MSE_loss(
                self.transform(y_hat, mean, std), self.transform(y, mean, std)
            )
            # loss += 0.5*self.MSE_loss(y_hat_diag, y_diag[:, rollout_step, :, :])
            loss += self.MSE_loss(y_hat_diag, y_diag_step)

        # scale loss
        loss *= 1.0 / CONFIG["roll_out"]
        if val:
            return (
                loss,
                x_state[0, 0, :, :],
                y[0, :, :],
                y_hat[0, :, :],
                y_diag_step[0, :, :],
                y_hat_diag[0, :, :],
            )
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch, batch_idx)
        self.log(
            "loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=train_batch[0].shape[0],
            sync_dist=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            loss, x0, targ, pred, targ_diag, pred_diag = self._step(
                val_batch, batch_idx, val=True
            )
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=val_batch[0].shape[0],
            sync_dist=True,
        )
        if ((self.current_epoch + 1) % CONFIG["logging"]["plot_freq"] == 0) & (
            batch_idx == 0
        ):
            self.log_fig_mlflow(x0, targ, pred, targ_diag, pred_diag)
        return loss

    @rank_zero_only
    def log_fig_mlflow(self, input, y, logits, y_diag, diag_logits):
        fig = make_map_val_plot(
            input.cpu().numpy(),
            y.cpu().numpy(),
            logits.cpu().numpy(),
            y_diag.cpu().numpy(),
            diag_logits.cpu().numpy(),
            self.ds.lats,
            self.ds.lons,
            self.ds.targ_lst,
            self.ds.targ_diag_lst,
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
        optimizer = torch.optim.Adam(self.parameters(), lr=CONFIG["lr"])
        return optimizer
