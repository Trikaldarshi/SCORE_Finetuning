# no model is needed for this expert
# empty definition

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, **kargs):
        super(Model, self).__init__()

    def forward(self, x):
        return x