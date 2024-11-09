import torch.nn as nn
import torch.nn.functional as F


class ChessNet(nn.Module):
    def __init__(self, metadata_size):
        super(ChessNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8 + metadata_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.output = nn.Linear(256, 1)

    def forward(self, x_board, x_meta):
        x = F.relu(self.bn1(self.conv1(x_board)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat([x, x_meta], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.output(x)
        return output.squeeze()
