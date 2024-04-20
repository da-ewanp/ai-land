import logging
import signal

import pytorch_lightning as pl
import torch
import yaml
from data_module import EcDataset, NonLinRegDataModule
from model import NonLinearRegression
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from train_callbacks import PlotCallback

# from pytorch_lightning.utilities.distributed import rank_zero_only
# from torch.distributed import init_process_group, destroy_process_group

logging.basicConfig(level=logging.INFO)


with open("config.yaml") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


if __name__ == "__main__":
    # Set device
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    data_module = NonLinRegDataModule()
    dataset = EcDataset()
    if CONFIG["logging"]["logger"] == "csv":
        logger = CSVLogger(
            CONFIG["logging"]["location"], name="testing"
        )  # Change 'logs' to the directory where you want to save the logs
    elif CONFIG["logging"]["logger"] == "mlflow":
        logger = MLFlowLogger(
            experiment_name=CONFIG["logging"]["project"],
            run_name=CONFIG["logging"]["name"],
            tracking_uri=CONFIG["logging"]["uri"],  # "file:./mlruns",
        )
    else:
        logger = None

    checkpoint_callback = ModelCheckpoint(monitor="val_R2", mode="max")

    # Setting a small validation dataset for plotting during training
    logging.info("Opening dataset for plotting...")
    test_ds = EcDataset(
        start_yr=2022,
        end_yr=2022,
        x_idxs=(500, 500 + 1),
        path="/ec/res4/hpcperm/daep/ecland_i6aj_o400_2010_2022_6hr_subset.zarr",
    )
    logging.info("Setting plot callback...")
    plot_callback = PlotCallback(10, test_ds, device=device, logger=logger)

    # print("Opening dataset for plotting...")
    # dataset_plot = EcDataset("2022", "2022", (8239, 8240))
    # print("Setting plot callback...")
    # plot_callback = PlotCallback(plot_frequency=1, dataset=dataset_plot)

    # train
    logging.info("Setting model params...")
    input_clim_dim = len(dataset.static_feat_lst)
    input_met_dim = len(dataset.dynamic_feat_lst)
    input_state_dim = len(dataset.targ_lst)
    output_dim = len(dataset.targ_lst)  # Number of output targets
    output_diag_dim = len(dataset.targ_diag_lst)
    hidden_dim = CONFIG["hidden_dim"]  # Number of hidden units
    model_pyt = NonLinearRegression(
        input_clim_dim,
        input_met_dim,
        input_state_dim,
        hidden_dim,
        output_dim,
        output_diag_dim,
    )

    torch.set_float32_matmul_precision("high")

    logging.info("Setting Trainer...")
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, RichProgressBar()],
        max_epochs=CONFIG["max_epochs"],
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        strategy=CONFIG["strategy"],
        devices=CONFIG["devices"],
        # barebones=True,
    )

    logging.info("Training...")
    trainer.logger.log_hyperparams(CONFIG)
    trainer.fit(model_pyt, data_module)

    logging.info("Saving model...")
    model_pyt.eval()
    torch.save(model_pyt.state_dict(), CONFIG["model_path"])
