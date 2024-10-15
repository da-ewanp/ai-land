import logging
import os

import mlflow
import pytorch_lightning as pl
import torch
import yaml
from data_module_norm import EcDataset, NonLinRegDataModule, Normalizer
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
from model_simple import NonLinearRegression
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

# from pytorch_lightning.plugins.environments import SLURMEnvironment
from train_callbacks import PlotCallback

# import signal


# from pytorch_lightning.utilities.distributed import rank_zero_only
# from torch.distributed import init_process_group, destroy_process_group

logging.basicConfig(level=logging.INFO)


PATH_NAME = os.path.dirname(os.path.abspath(__file__))
with open(f"{PATH_NAME}/config.yaml") as stream:
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

    normalizer = Normalizer(
        path=CONFIG["file_path"], normalize_dict=CONFIG["normalize"]
    )
    data_module = NonLinRegDataModule()
    dataset = EcDataset(normalizer=normalizer)

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
        logging.info(f"logging to {logger.run_id}")
    else:
        logger = None

    # print(torch.version.cuda)

    # checkpoint_callback = ModelCheckpoint(monitor="val_R2", mode="max")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    # checkpoint_callback = ModelCheckpoint(monitor="val_step_loss", mode="min")

    # Setting a small validation dataset for plotting during training
    logging.info("Opening dataset for plotting...")
    test_ds = EcDataset(
        start_yr=2022,
        end_yr=2022,
        x_idxs=(500, 500 + 1),
        # path="/ec/res4/hpcperm/daep/ecland_i6aj_o400_2010_2022_6hr_subset.zarr",
        path="/hpcperm/daep/ecland_i6aj_2017_2022_1h_subset.zarr",
        # x_idxs=(9973, 9973 + 1),
        # # path="/ec/res4/scratch/daep/ec_training_db_out_O200/ecland_i8ki_2010_2022_6h.zarr"
        # path="/ec/res4/hpcperm/daep/ec_land_training_db/ecland_i8ki_o200_2010_2022_6h.zarr",
        normalizer=normalizer,
    )
    logging.info("Setting plot callback...")
    plot_callback = PlotCallback(
        plot_frequency=CONFIG["logging"]["plot_freq"],
        dataset=test_ds,
        device=device,
        # device="cpu",
        logger=logger,
    )

    # print("Opening dataset for plotting...")
    # dataset_plot = EcDataset("2022", "2022", (8239, 8240))
    # print("Setting plot callback...")
    # plot_callback = PlotCallback(plot_frequency=1,
    #                              dataset=dataset_plot,
    #                              device=device,
    #                              logger=logger)

    # std = dataset.y_prog_stdevs.cpu().numpy()
    std = dataset.normalizer.y_prog_stdevs.cpu().numpy()
    # std = dataset.y_prog_stdevs_test.cpu().numpy()

    # ds_mean = np.nanmean(dataset.ds_ecland.firstdiff_means[slice(*dataset.x_idxs), dataset.targ_index], axis=0) / (std + 1e-5)
    # ds_std =dataset.ds_ecland.firstdiff_stdevs[slice(*dataset.x_idxs), dataset.targ_index] / (std + 1e-5)
    # ds_std = np.nanstd(dataset.ds_ecland.firstdiff_stdevs[slice(*dataset.x_idxs), dataset.targ_index], axis=0) / (std + 1e-5)
    # ds_mean = dataset.ds_ecland.firstdiff_means[dataset.targ_index] / std  # (std + 1e-5)
    # ds_std = dataset.ds_ecland.firstdiff_stdevs[dataset.targ_index] / std  # (std + 1e-5)
    ds_mean = dataset.ds_ecland.data_1stdiff_means[dataset.targ_index] / std
    ds_std = dataset.ds_ecland.data_1stdiff_stdevs[dataset.targ_index] / std

    # ds_mean = dataset.ds_ecland.data_1stdiff_means[dataset.targ_index] / (std + 1e-5)  # *** This is the one that works best
    # ds_std = dataset.ds_ecland.data_1stdiff_stdevs[dataset.targ_index] / (std + 1e-5)  # *** This is the one that works best
    # ds_mean = dataset.ds_ecland.data_1stdiff_means[dataset.targ_index]
    # ds_std = dataset.ds_ecland.data_1stdiff_stdevs[dataset.targ_index]

    # train
    logging.info("Setting model params...")
    input_clim_dim = len(dataset.clim_index)  # dataset.x_static_scaled.shape[-1]
    input_met_dim = len(dataset.dynamic_feat_lst)
    input_state_dim = len(dataset.targ_lst)
    output_dim = len(dataset.targ_lst)  # Number of output targets
    output_diag_dim = len(dataset.targ_diag_lst)
    hidden_dim = CONFIG["hidden_dim"]  # Number of hidden units

    upper_bound_indices = (
        None  # [dataset.targ_lst.index(x) for x in ["tsn", "snowc_recalc", "log10_sd"]]
    )
    lower_bound_indices = (
        None  # [dataset.targ_lst.index(x) for x in ["snowc_recalc", "sd"]]
    )

    # This worked!!!
    # lower_bound_indices_prog = [dataset.targ_lst.index(x) for x in ["sd"]]
    # lower_bound_indices_diag = None # [dataset.targ_diag_lst.index(x) for x in ["snowc_recalc"]]
    # upper_lower_bound_indices_prog = [dataset.targ_lst.index(x) for x in ["rsn", "tsn",]]
    # upper_lower_bound_indices_diag = [dataset.targ_diag_lst.index(x) for x in ["snowc_recalc"]]

    lower_bound_indices_prog = [dataset.targ_lst.index(x) for x in ["sd"]]
    lower_bound_indices_diag = (
        None  # [dataset.targ_diag_lst.index(x) for x in ["snowc_recalc"]]
    )
    upper_bound_indices_prog = [dataset.targ_lst.index(x) for x in ["tsn"]]
    upper_bound_indices_diag = (
        None  # [dataset.targ_diag_lst.index(x) for x in ["snowc_recalc"]]
    )
    upper_lower_bound_indices_prog = (
        None  # [dataset.targ_lst.index(x) for x in ["rsn"]]
    )
    upper_lower_bound_indices_diag = [
        dataset.targ_diag_lst.index(x) for x in ["snowc_recalc"]
    ]

    model_pyt = NonLinearRegression(
        input_clim_dim,
        input_met_dim,
        input_state_dim,
        hidden_dim,
        output_dim,
        output_diag_dim,
        mu_norm=ds_mean,
        std_norm=ds_std,
        dataset=dataset,
        lower_bounded_indices_prog=lower_bound_indices_prog,
        lower_bounded_indices_diag=lower_bound_indices_diag,
        upper_bounded_indices_prog=upper_bound_indices_prog,
        upper_bounded_indices_diag=upper_bound_indices_diag,
        upper_lower_bounded_indices_prog=upper_lower_bound_indices_prog,
        upper_lower_bounded_indices_diag=upper_lower_bound_indices_diag,
    )
    # mod_path = "/home/daep/projects/ai-land/ai_land_minimal/mlruns/791953591827285582/34aa9b355b1e4c8fa6004d56136daf0a/checkpoints/epoch=3-step=14608.ckpt"
    # mod_path = "/home/daep/projects/ai-land/ai_land_minimal/mlruns/791953591827285582/adc9002c49dc49e3b383dda14bd0b3e3/checkpoints/epoch=69-step=140000.ckpt"
    # mod_path = "/home/daep/projects/ai-land/ai_land_minimal/mlruns/791953591827285582/dc2bb5b6949b46f8b764c607f3a78b53/checkpoints/epoch=23-step=87648.ckpt"
    # mod_path = "/home/daep/projects/ai-land/ai_land_minimal/mlruns/791953591827285582/247906d022bb4cddb8da89f0417a4ee2/checkpoints/epoch=47-step=96000.ckpt"
    # mod_path = "/home/daep/projects/ai-land/ai_land_minimal/mlruns/791953591827285582/394d0179061e46c8af7acbaf03449c9b/checkpoints/epoch=24-step=50000.ckpt"
    mod_path = "/home/daep/projects/ai-land/ai_land_minimal/mlruns/791953591827285582/7784c5101ea64e25b8da6e7e2ac58b2a/checkpoints/epoch=37-step=76000.ckpt"
    model_pyt.load_state_dict(torch.load(mod_path)["state_dict"])

    # torch.set_float32_matmul_precision("high")
    torch.set_float32_matmul_precision("medium")

    logging.info("Setting Trainer...")
    metrics_monitor = SystemMetricsMonitor(
        run_id=logger.run_id,
        sampling_interval=20,  # Log every 5 seconds
    )
    metrics_monitor.start()

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, plot_callback],
        # callbacks=[checkpoint_callback, RichProgressBar()],
        # callbacks=[checkpoint_callback],
        max_epochs=CONFIG["max_epochs"],
        # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        strategy=CONFIG["strategy"],
        devices=CONFIG["devices"],
        # barebones=True,
        limit_train_batches=CONFIG["limit_train_batches"],
        limit_val_batches=CONFIG["limit_val_batches"],
        # gradient_clip_val=CONFIG["gradient_clip_val"],
        # gradient_clip_algorithm=CONFIG["gradient_clip_algorithm"],
    )

    logging.info("Training...")
    trainer.logger.log_hyperparams(CONFIG)
    trainer.fit(model_pyt, data_module)
    # metrics_monitor.stop()

    logging.info("Saving model...")
    model_pyt.eval()
    torch.save(model_pyt.state_dict(), CONFIG["model_path"])
