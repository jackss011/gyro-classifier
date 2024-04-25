import torch.nn as nn
from binary_modules import BinarizeLinear, BinarizeConv2d


class CNN_binary(nn.Module):
    def __init__(self, numClasses):
        super(CNN_binary, self).__init__()
        self.net = nn.Sequential(
            # First layer: Convolutional layer with kernel=[1,9], stride=[0,4]: form 1*6*128 -> 32*6*64
            BinarizeConv2d(1, 32, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(inplace=True),

            # Second layer: Pool layer with kernel=[1,2], stride=[1,2]: form 32*6*64 -> 32*6*32
            nn.MaxPool2d(kernel_size=[1, 2], stride=(1, 2)),

            # Third layer: Convolutional layer with kernel=[1,3], stride=1: form 32*6*32 -> 64*6*32
            BinarizeConv2d(32, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),

            # Fourth Layer
            BinarizeConv2d(64, 128, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),

            # Fifth layer: Pool layer with kernel=[1,2], stride=[1,2]: form 128*6*32 -> 128*6*16
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            BinarizeConv2d(128, 128, kernel_size=(6, 1), stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True)
        )

        self.fc = BinarizeLinear(2048, numClasses)

    def forward(self, x):
        # forward operation: propagates the input though the the layers predicting the result
        out = self.net(x)
        out = out.reshape(out.size(0), -1)  # flat the matrix into an array
        out = self.fc(out)
        return out


class CNN_binary_notlast(nn.Module):
    def __init__(self, numClasses):
        super(CNN_binary_notlast, self).__init__()
        self.net = nn.Sequential(
            # First layer: Convolutional layer with kernel=[1,9], stride=[0,4]: form 1*6*128 -> 32*6*64
            # nn.Conv2d(1, 32, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4)),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            BinarizeConv2d(1, 32, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(inplace=True),

            # Second layer: Pool layer with kernel=[1,2], stride=[1,2]: form 32*6*64 -> 32*6*32
            nn.MaxPool2d(kernel_size=[1, 2], stride=(1, 2)),

            # Third layer: Convolutional layer with kernel=[1,3], stride=1: form 32*6*32 -> 64*6*32
            # nn.Conv2d(32, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            BinarizeConv2d(32, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),

            # Fourth Layer
            # nn.Conv2d(64, 128, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            BinarizeConv2d(64, 128, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),

            # Fifth layer: Pool layer with kernel=[1,2], stride=[1,2]: form 128*6*32 -> 128*6*16
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            # Sixth layer: Convolutional layer with kernel=[6,1], stride=1: form 128*6*16 -> 128*1*16
            # nn.Conv2d(128, 128, kernel_size=(6, 1), stride=1, padding=0),
            # nn.BatchNorm2d(128),
            # nn.ReLU()
            BinarizeConv2d(128, 128, kernel_size=(6, 1), stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True)
        )

        self.fc = nn.Linear(2048, numClasses)
        # self.fc = BinarizeLinear(2048, numClasses)

    def forward(self, x):
        # forward operation: propagates the input though the the layers predicting the result
        out = self.net(x)
        out = out.reshape(out.size(0), -1)  # flat the matrix into an array
        out = self.fc(out)
        return out
