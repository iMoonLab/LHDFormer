from omegaconf import DictConfig
from ..utils.device import device_set
from .LHDFormer import LHDFormer
device = device_set()

def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config).to(device)
