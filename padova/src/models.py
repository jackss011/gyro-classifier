import torch.nn as nn
import collections


class CNN(nn.Module):
    def __init__(self, numClasses):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            collections.OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 32, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4))),
                    ("bn1", nn.BatchNorm2d(32)),
                    #("htanh1", nn.Hardtanh()),
                    ("htanh1", nn.ReLU()),
                    ("maxpool1", nn.MaxPool2d(kernel_size=[1, 2], stride=(1, 2))),
                    ("conv2", nn.Conv2d(32, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))),
                    ("bn2", nn.BatchNorm2d(64)),
                    #("htanh2", nn.Hardtanh()),
                    ("htanh2", nn.ReLU()),
                    ("conv3", nn.Conv2d(64, 128, kernel_size=(1, 3), stride=1, padding=(0, 1))),
                    ("bn3", nn.BatchNorm2d(128)),
                    #("htanh3", nn.Hardtanh()),
                    ("htanh3", nn.ReLU()),
                    ("maxpool2", nn.MaxPool2d(kernel_size=[1, 2], stride=(1, 2))),
                    ("conv4", nn.Conv2d(128, 128, kernel_size=(6, 1), stride=1, padding=0)),
                    ("bn4", nn.BatchNorm2d(128)),
                    #("htanh4", nn.Hardtanh()),
                    ("htanh4", nn.ReLU()),
                ]
            )
        )
        self.fc = nn.Linear(2048, numClasses)

    def forward(self, x):
        out = self.net(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
