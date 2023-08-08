import hydra
from omegaconf import DictConfig
import datetime
import torch
from pathlib import Path
import logging

from tqdm import tqdm

def logger_init(logger_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    Path('test_logs').mkdir(parents=True, exist_ok=True)

    log_file = f'test_logs/{logger_name}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


## Here we use the same loss function and stat methods given in the official LocUNet repo

def official_loss(output, target):
    loss = torch.sum((output - target)**2,1)
    loss = torch.sqrt(loss)
    loss = torch.mean(loss)
    return loss

def calc_loss_test(pred, target):
    loss = official_loss(pred, target)# *256*256
    return loss

@hydra.main(version_base=None, config_path="configs", config_name="RSS_only_final_noise.yaml")
# @hydra.main(version_base=None)
def eval(cfg : DictConfig) -> None:
    dstr = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    exp_name = cfg.experiment_name

    logger_name = f"test-{exp_name}-{dstr}"
    logger = logger_init(logger_name)


    if cfg.gpus:
        device = torch.device('cuda:{}'.format(cfg.gpus[0]))
    else:
        device = torch.device('cpu')

    val_dataloader = hydra.utils.instantiate(cfg.val_dataloader)
    model = hydra.utils.instantiate(cfg.model).to(device)


    assert cfg.resume_path is not None, "resume_path must be provided for validation!!!!!"
    ckpt = torch.load(cfg.resume_path)
    logger.info(f"Loading checkpoint from {Path(cfg.resume_path).stem}")
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    loss_total = 0
    epoch_samples = 0
    
    logger.info("Start test")

    for index, data in enumerate(tqdm(val_dataloader)):
        inputs = (data['feat']*2.-1.).to(device)
        targets = data['gt'].to(device)
        outputs1 = model(inputs) * 255.
        loss = calc_loss_test(outputs1.float(), targets.float())
        logger.info(f"loss for batch index-{index}: {loss.item()}")
        loss_total += loss.item() * targets.size(0)
        epoch_samples += inputs.size(0)
    
    logger.info("Test finished")
    logger.info("**************************************")
    logger.info(f"Average loss: {loss_total / epoch_samples}")


if __name__ == "__main__":
    with torch.no_grad():
        eval()