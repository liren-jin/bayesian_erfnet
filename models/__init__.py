import torch
import yaml
from constants import Models
from models.models import ERFNet, AleatoricERFNet
from pytorch_lightning.core.lightning import LightningModule


def get_model(cfg) -> LightningModule:
    name = cfg["model"]["name"]
    if isinstance(cfg, dict):
        if name == Models.ERFNET:
            return ERFNet(
                cfg,
            )
        elif name == Models.BAYESIAN_ERFNET:
            return AleatoricERFNet(
                cfg,
            )
        else:
            RuntimeError(f"{name} model not implemented")
    else:
        raise RuntimeError(f"{type(cfg)} not a valid config")


def load_pretrained_model(config_path: str, checkpoint_path: str) -> LightningModule:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(config_path, "r") as config_file:
        cfg = yaml.safe_load(config_file)

    return get_model(cfg).load_from_checkpoint(checkpoint_path, cfg=cfg).to(device)


# def get_criterion(cfg):
#   name = cfg['model']['loss']
#   if name == "xentropy":
#     return CrossEntropyLoss()
#   else:
#     raise RuntimeError("Loss {} not available".format(name))

####################
# for testing only #
####################


# def get_net(cfg):
#     num_classes = cfg["model"]["num_classes"]
#     in_channels = cfg["model"]["in_channels"]
#     name = cfg["model"]["name"]
#     if (
#         type(num_classes) == int
#         and num_classes > 0
#         and type(in_channels) == int
#         and in_channels > 0
#     ):
#         if name == "erfnet":
#             return ERFNetModel(num_classes, in_channels)
#         else:
#             RuntimeError("{} model not implemented".format(name))
#     else:
#         raise RuntimeError("{} not a valid config".format(type(cfg)))
