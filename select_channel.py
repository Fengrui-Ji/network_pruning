import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
import math
import numpy as np


def get_channel_L2(Conv_weight):
    channel_norms = torch.sum(Conv_weight.weight.data.abs(),dim=(1,2, 3)) 
    return (channel_norms - channel_norms.min()) / (channel_norms.max() - channel_norms.min())

