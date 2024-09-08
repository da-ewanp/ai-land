import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import memory
from pytorch_lightning.utilities import rank_zero_only

# Define the config for the experiment
PATH_NAME = os.path.dirname(os.path.abspath(__file__))
with open(f"{PATH_NAME}/config.yaml") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


class GPUMemoryCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            for i in range(trainer.num_devices):
                # Get memory usage for the current GPU
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)  # Convert to MB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)  # Convert to MB
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)  # Convert to MB
                
                memory_percentage = (memory_allocated / memory_total) * 100
                
                # Log metrics
                pl_module.log(f'GPU {i} Memory Allocated mb', memory_allocated, sync_dist=True)
                pl_module.log(f'GPU {i} Memory Reserved mb', memory_reserved, sync_dist=True)
                pl_module.log(f'GPU {i} Memory Usage Percent', memory_percentage, sync_dist=True)


class PlotCallback(Callback):
    def __init__(self, plot_frequency, dataset, device, logger=None):
        super().__init__()
        self.device = device
        self.plot_frequency = plot_frequency
        self.test_ds = dataset
        self.clim, self.met, self.state, self.state_diag = self.test_ds.load_data()
        self.times = self.test_ds.times
        self.logger = logger
        self.x = self.test_ds.x_idxs[0]

    def ailand_plot(self, x_vals, preds, targs, label, ax):
        ax.plot(x_vals, targs, label="ec-land")
        ax.plot(x_vals, preds, "--", label="ai-land")
        ax.set_xlim(x_vals[[0, -1]])
        ax.set_xlabel("time")
        ax.set_title(label)
        ax.legend()
        return ax

    # def make_subplot(self, pl_module, epoch):
    def make_subplot(self, preds, targs, name_lst, epoch, plot_str):
        # Plotting code below
        fig, axes = plt.subplots(
            nrows=3,
            ncols=int(np.ceil(len(name_lst) / 3)),
            figsize=(16, 8),
        )
        for i, ax in enumerate(axes.flatten()):
            if i < len(name_lst):
                self.ailand_plot(
                    self.times[:],
                    preds[:, 0, i].cpu().numpy(),
                    targs[:, 0, i].cpu().numpy(),
                    name_lst[i],
                    ax,
                )
            else:
                ax.set_axis_off()
        fig.tight_layout()
        fig.autofmt_xdate()
        if self.logger is not None:
            if CONFIG["logging"]["logger"] == "mlflow":
                self.logger.experiment.log_figure(
                    self.logger.run_id,
                    fig,
                    f"timeseries{self.x}_{plot_str}_epoch{epoch + 1}.png",
                )
            else:
                fig.savefig(
                    f"{CONFIG['logging']['location']}/plots/timeseries{self.x}_{plot_str}_epoch{epoch + 1}.png"
                )
        plt.close(fig)

    def run_model(self, pl_module):
        pl_module.eval().to(self.device)
        with torch.no_grad():
            preds, diags = pl_module.predict_step(
                self.clim.to(self.device),
                self.met.to(self.device),
                self.state.to(self.device),
                self.state_diag.to(self.device),
            )
        return preds, diags

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_frequency == 0:
            preds, diags = self.run_model(pl_module)
            # Generate plot using the model's current weights
            self.make_subplot(preds, self.state, self.test_ds.targ_lst, epoch, "progs")
            self.make_subplot(
                diags, self.state_diag, self.test_ds.targ_diag_lst, epoch, "diags"
            )
