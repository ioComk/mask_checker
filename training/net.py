import torch as t
from torch import nn
from torch.nn.modules.activation import ReLU

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.act = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )

        self.l1 = nn.Sequential(
            nn.Linear(1600,512),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        # self.l2 = nn.Sequential(
        #     nn.Linear(4096, 4096),
        #     nn.Dropout(p=0.5),
        #     nn.ReLU()
        # )

        self.out = nn.Sequential(
            nn.Linear(512, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, input):

        x = self.conv1(input)       
        x = self.conv2(x)
        # x = self.conv3(x)

        x = self.flatten(x)

        x = self.l1(x)
        # x = self.l2(x)
        y = self.out(x)

        return y

    def flatten(self, x):
        bs = x.size()[0]
        return x.view(bs, -1)