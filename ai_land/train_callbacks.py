import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

# Define the config for the experiment
with open("config.yaml") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


class PlotCallback(Callback):
    def __init__(self, plot_frequency, dataset, device, logger=None):
        super().__init__()
        self.device = device
        self.plot_frequency = plot_frequency
        self.test_ds = dataset
        self.clim, self.met, self.state, self.state_diag = self.test_ds.load_data()
        self.times = self.test_ds.times
        self.logger = logger

    def ailand_plot(self, x_vals, preds, targs, label, ax):
        ax.plot(x_vals, targs, label="ec-land")
        ax.plot(x_vals, preds, "--", label="ai-land")
        ax.set_xlim(x_vals[[0, -1]])
        ax.set_xlabel("time")
        ax.set_title(label)
        ax.legend()
        return ax

    def make_subplot(self, pl_module, epoch):
        with torch.no_grad():
            preds, _ = pl_module.predict_step(
                self.clim.to(self.device),
                self.met.to(self.device),
                self.state.to(self.device),
                self.state_diag.to(self.device),
            )
            # Assuming output is what you need for plotting
            # Plotting code below
            fig, axes = plt.subplots(
                nrows=3,
                ncols=int(np.ceil(len(self.test_ds.targ_lst) / 3)),
                figsize=(16, 8),
            )
            for i, ax in enumerate(axes.flatten()):
                if i < len(self.test_ds.targ_lst):
                    self.ailand_plot(
                        self.times[:],
                        preds[:, 0, i].cpu().numpy(),
                        self.state[:, 0, i].cpu().numpy(),
                        self.test_ds.targ_lst[i],
                        ax,
                    )
                else:
                    ax.set_axis_off()
            fig.tight_layout()
            fig.autofmt_xdate()
            if self.logger is not None:
                if CONFIG["logging"]["logger"] == "mlflow":
                    self.logger.experiment.log_figure(
                        self.logger.run_id, fig, f"timeseries_epoch{epoch + 1}.png"
                    )
                else:
                    fig.savefig(
                        f"{CONFIG['logging']['location']}/plots/timeseries_epoch{epoch + 1}.png"
                    )
            plt.close()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_frequency == 0:
            pl_module.eval().to(self.device)
            # Generate plot using the model's current weights
            self.make_subplot(pl_module, epoch)
