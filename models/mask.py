import torch
import torch.nn as nn


class MaskLayer(nn.Module):
    def __init__(self):
        super(MaskLayer, self).__init__()

    def forward(self, x, mask):
        neg_inf = -1e9
        masked_x = torch.where(mask == 1, x, neg_inf)
        return masked_x
