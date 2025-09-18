import torch

def device_set():
    device = torch.device("cuda:4")
    # device = torch.device("cuda:4")
    return device