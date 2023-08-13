from datasets.cityscape import CityscapesDataModule
from datasets.flightmare import FlightmareDataModule
from datasets.potsdam import PotsdamDataModule
from datasets.shapenet import ShapenetDataModule
from pytorch_lightning import LightningDataModule


def get_data_module(cfg) -> LightningDataModule:
    name = cfg["data"]["name"]
    if name == "shapenet":
        return ShapenetDataModule(cfg)
    elif name == "cityscapes":
        return CityscapesDataModule(cfg)
    elif name == "potsdam":
        return PotsdamDataModule(cfg)
    elif name == "flightmare":
        return FlightmareDataModule(cfg)
    else:
        raise ValueError(f"Dataset '{name}' not found!")
