import pytorch_lightning as pl
import torch
import wandb
import yaml
from data_module import EcDataset
from data_module import NonLinRegDataModule
from model import NonLinearRegression
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from train_callbacks import PlotCallback

# from model_simple import NonLinearRegression


with open("config.yaml") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Set device
if torch.cuda.is_available():
    DEV = "cuda:0"
else:
    DEV = "cpu"
device = torch.device(DEV)
print(device)

data_module = NonLinRegDataModule()
dataset = EcDataset()
dataset_test = EcDataset(2022, 2022, (8239, 8240))

if CONFIG["logging"]["logger"] == "csv":
    logger = CSVLogger("logs", name=CONFIG["logging"]["name"])  # Change 'logs' to the directory where you want to save the logs
elif CONFIG["logging"]["logger"] == "wandb":
    logger = WandbLogger(name=CONFIG["logging"]["name"], project=CONFIG["logging"]["project"])

checkpoint_callback = ModelCheckpoint(monitor="val_r**2", mode="max")
plot_callback = PlotCallback(plot_frequency=20, dataset=dataset_test)

# Set dims
input_dim = len(dataset.dynamic_feat_lst + dataset.static_feat_lst)  # Number of input features
input_clim_dim = len(dataset.static_feat_lst)
input_met_dim = len(dataset.dynamic_feat_lst)
input_state_dim = len(dataset.targ_lst)
output_dim = len(dataset.targ_lst)
hidden_dim = 80  # 172  # 128  # Number of hidden units

# Set model
model_pyt = NonLinearRegression(
    input_clim_dim,
    input_met_dim,
    input_state_dim,
    hidden_dim,
    output_dim,
)

torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")

# train
trainer = pl.Trainer(
    logger=logger,
    callbacks=[checkpoint_callback, RichProgressBar()],
    max_epochs=80,  # 40  # 100,  # 200,
    distributed_backend="ddp",
    # enable_progress_bar = False,
    # log_every_n_steps=490,
    # barebones=True,
    # devices=1, accelerator="gpu"
    # precision='bf16-mixed',
    # logger=csv_logger,
)

wandb.login()
trainer.fit(model_pyt, data_module)
wandb.finish()

model_pyt.eval()
torch.save(model_pyt.state_dict(), CONFIG["model_path"])
