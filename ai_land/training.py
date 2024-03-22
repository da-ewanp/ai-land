import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger

import wandb
from data_module import EcDataset, NonLinRegDataModule
from model_simple import NonLinearRegression


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
# csv_logger = CSVLogger('logs', name='testing')  # Change 'logs' to the directory where you want to save the logs
wandb_logger = WandbLogger(name=CONFIG["logging"]["name"], project=CONFIG["logging"]["project"])
checkpoint_callback = ModelCheckpoint(monitor="val_r**2", mode="max")

# train
input_dim = len(
    dataset.dynamic_feat_lst + dataset.static_feat_lst
)  # Number of input features
output_dim = len(dataset.targ_lst)  # Number of output targets
hidden_dim = 172  # 128  # Number of hidden units
model_pyt = NonLinearRegression(input_dim, output_dim, hidden_dim, DEV)

torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")

trainer = pl.Trainer(
    logger=wandb_logger,
    callbacks=[checkpoint_callback, RichProgressBar()],
    max_epochs=80,  # 40  # 100,  # 200,
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
