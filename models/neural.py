import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mask import MaskLayer
from utils.constants import ACTION_SPACE_SIZE


class ChessNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, ACTION_SPACE_SIZE)
        self.mask_layer = MaskLayer()

    def forward(self, x, mask):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        masked_logits = self.mask_layer(logits, mask)
        return masked_logits
