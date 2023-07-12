import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# @hydra.main(version_base=None, config_path="configs", config_name="RSS_noisy_ToA_without_citymap")
@hydra.main(version_base=None)
def main(cfg : DictConfig) -> None:
    train_dataloader = hydra.utils.instantiate(cfg.train_dataloader)
    val_dataloader = hydra.utils.instantiate(cfg.val_dataloader)
    model = hydra.utils.instantiate(cfg.model)
    dstr = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    path = f'logs/swinunet-{cfg.experiment_name}-{dstr}'

    tb_logger = pl_loggers.TensorBoardLogger(path)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor="val_loss",
        mode="min",
        dirpath=path,
        filename=f"swinunet-{cfg.experiment_name}"+"-{epoch:02d}-{val_loss:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    if cfg.gpus:
        trainer = pl.Trainer(gpus=cfg.gpus, precision=16, logger=tb_logger, callbacks=[checkpoint_callback,lr_monitor],resume_from_checkpoint=cfg.resume_path)
    else:
        trainer = pl.Trainer(gpus=4, precision=16, logger=tb_logger, callbacks=[checkpoint_callback,lr_monitor],resume_from_checkpoint=cfg.resume_path)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == "__main__":
    main()