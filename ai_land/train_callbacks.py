import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from pytorch_lightning.callbacks import Callback
from torch import tensor


with open("config.yaml") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


class PlotCallback(Callback):
    def __init__(self, plot_frequency, dataset):
        super().__init__()
        self.plot_frequency = plot_frequency
        self.test_ds = dataset
        self.static_feats = self.test_ds.X_static_scaled[0, 0, :]
        self.feats = self.test_ds.dynamic_feat_scalar.transform(
            tensor(
                self.test_ds.ds_ecland[self.test_ds.dynamic_feat_lst].isel(x=0, time=slice(0, -1)).compute().to_array().values.T,
                dtype=torch.float32,
            )
        )
        self.states = self.test_ds.targ_scalar.transform(
            tensor(
                self.test_ds.ds_ecland[self.test_ds.targ_lst].isel(x=0, time=slice(0, -1)).compute().to_array().values.T,
                dtype=torch.float32,
            )
        )

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
            preds = pl_module.predict(self.static_feats, self.feats, self.states)
            # Assuming output is what you need for plotting
            # Plotting code example
            fig, axes = plt.subplots(nrows=3, ncols=int(np.ceil(len(self.test_ds.targ_lst) / 3)), figsize=(16, 8))
            for i, ax in enumerate(axes.flatten()):
                if i < len(self.test_ds.targ_lst):
                    self.ailand_plot(
                        self.test_ds.ds_ecland.time.values[:-1],
                        preds[:, i].cpu().numpy(),
                        self.states[:, i].cpu().numpy(),
                        self.test_ds.targ_lst[i],
                        ax,
                    )
                else:
                    ax.set_axis_off()
            fig.tight_layout()
            fig.autofmt_xdate()
            fig.savefig(f"logs/plots/test_epoch{epoch + 1}.png")
            if CONFIG["logging"]["logger"] == "wandb":
                wandb.log({"time-series": wandb.Image(fig)})
            plt.close()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_frequency == 0:
            # Generate plot using the model's current weights
            self.make_subplot(pl_module, epoch)
