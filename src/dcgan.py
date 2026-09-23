import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)


# Define the Generator Network
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()
        nc, nz, ngf = params['nc'], params['nz'], params['ngf']

        # Input Dimension: (nc = 1) x 1 x 100
        # CN1: 100 input channel, 256 output channels, 4x4 square convolution, stride=2, padding=0
        self.conv1 = nn.ConvTranspose2d(nz, ngf * 4, kernel_size = 4, stride = 2, padding = 0)
        # CN2: 256 input channel, 128 output channels, 4x4 square convolution, stride=2, padding=1
        self.conv2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size = 4, stride = 2, padding = 1)
        # CN3: 128 input channel, 64 output channels, 4x4 square convolution, stride=2, padding=1
        self.conv3 = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size = 4, stride = 2, padding = 1)
        # CN4: 64 input channel, 1 output channel, 4x4 square convolution, stride=2, padding=1
        self.conv4 = nn.ConvTranspose2d(ngf, nc, kernel_size = 4, stride = 2, padding = 1)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(ngf * 4)
        self.bn2 = nn.BatchNorm2d(ngf * 2)
        self.bn3 = nn.BatchNorm2d(ngf)


    def forward(self, input):
        # Convolution layer C1: 1 input channel, 64 output channels,
        # 4x4 square convolution, it uses Batch normalization and RELU activation function, and
        # outputs a Tensor with size (N, 64, 16, 16), where N is the size of the batch
        c1 = F.relu(self.bn1(self.conv1(input)))
        
        # Convolution layer C2: 64 input channel, 128 output channels,
        # 4x4 square convolution, it uses Batch normalization and RELU activation function, and        
        # outputs a Tensor with size (N, 128, 8, 8), where N is the size of the batch
        c2 = F.relu(self.bn2(self.conv2(c1)))
                
        # Convolution layer C3: 128 input channel, 256 output channel
        # 4x4 square convolution, it uses Batch normalization and RELU activation function, and
        # outputs a Tensor with size (N, 256, 4, 4), where N is the size of the batch
        c3 = F.relu(self.bn3(self.conv3(c2)))
        
        # Convolution layer C4: 256 input channel, 1 output channel
        # 4x4 square convolution, it uses Sigmoid activation function, and
        # outputs a Tensor with size (N, 1, 1, 1), where N is the size of the batch
        output = F.tanh(self.conv4(c3))
                
        return output


class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        nc, ndf = params['nc'], params['ndf']

        # Input Dimension: (nc = 1) x 32 x 32
        # CN1: 1 input channel, 64 output channels, 4x4 square convolution, stride=2, padding=1
        self.conv1 = nn.Conv2d(nc, ndf, kernel_size = 4, stride = 2, padding = 1)
        # CN2: 64 input channel, 128 output channels, 4x4 square convolution, stride=2, padding=1
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size = 4, stride = 2, padding = 1)
        # CN3: 128 input channel, 256 output channels, 4x4 square convolution, stride=2, padding=1
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size = 4, stride = 2, padding = 1)
        # CN4: 256 input channel, 1 output channel, 4x4 square convolution, stride=2, padding=0
        self.conv4 = nn.Conv2d(ndf * 4, nc, kernel_size = 4, stride = 2, padding = 0)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
                

    def forward(self, input):
        # Convolution layer C1: 1 input channel, 64 output channels,
        # 4x4 square convolution, it uses Batch normalization and RELU activation function, and
        # outputs a Tensor with size (N, 64, 16, 16), where N is the size of the batch
        c1 = F.relu(self.bn1(self.conv1(input)))

        # Convolution layer C2: 64 input channel, 128 output channels,
        # 4x4 square convolution, it uses Batch normalization and RELU activation function, and        
        # outputs a Tensor with size (N, 128, 8, 8), where N is the size of the batch
        c2 = F.relu(self.bn2(self.conv2(c1)))
        
        # Convolution layer C3: 128 input channel, 256 output channel
        # 4x4 square convolution, it uses Batch normalization and RELU activation function, and
        # outputs a Tensor with size (N, 256, 4, 4), where N is the size of the batch
        c3 = F.relu(self.bn3(self.conv3(c2)))

        # Convolution layer C4: 256 input channel, 1 output channel
        # 4x4 square convolution, it uses Sigmoid activation function, and
        # outputs a Tensor with size (N, 1, 1, 1), where N is the size of the batch
        output = F.sigmoid(self.conv4(c3))
        
        return output