import numpy as np
from torch import nn

KERNEL_SIZE = 3
STRIDE = 2
padding = 0
dilation = 1

class Net(nn.Module):

    def __init__(self, in_c, h, w, mid_c, out_c, hidden_units, out_units, dropout=0):
        super(Net, self).__init__()

        p = dropout

        h_out1, w_out1 = self.calc_cnn_shape(h, w, padding, dilation, KERNEL_SIZE, STRIDE)
        h_out2, w_out2 = self.calc_cnn_shape(h_out1, w_out1, padding, dilation, KERNEL_SIZE, STRIDE)
        h_out3, w_out3 = self.calc_cnn_shape(h_out2, w_out2, padding, dilation, KERNEL_SIZE, STRIDE)
        h_out4, w_out4 = self.calc_cnn_shape(h_out3, w_out3, padding, dilation, KERNEL_SIZE, STRIDE)

        in_unit = out_c*h_out4*w_out4

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=padding, dilation=dilation),
            nn.Conv2d(mid_c, out_c, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=padding, dilation=dilation)
        )

        self.dense = nn.Sequential(
            nn.Linear(in_unit, hidden_units),
            nn.Dropout(p),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(p),
            nn.ReLU(),
            nn.Linear(hidden_units, out_units),
            nn.Dropout(p),
            nn.ReLU(),
            nn.Linear(out_units, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, input):

        x = self.conv(input)
        x = self.flatten(x)
        y = self.dense(x)

        return y

    def flatten(self, x):
        bs = x.size()[0]
        return x.view(bs, -1)
    
    def calc_cnn_shape(self, h_in, w_in, padding, dilation, kernel_size, stride):

        h_out = int(np.floor(((h_in+(2*padding)-(dilation*(kernel_size-1))-1) / stride) + 1))
        w_out = int(np.floor(((w_in+(2*padding)-(dilation*(kernel_size-1))-1) / stride) + 1))

        return h_out, w_out