import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_normal(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

#Generator Network i.e. G(z)
class Generator(nn.Module):

    def __init__(self, dim=128):
        super(Generator, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(100, dim*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(dim*8)
        self.deconv2 = nn.ConvTranspose2d(dim*8, dim*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(dim*4)
        self.deconv3 = nn.ConvTranspose2d(dim*4, dim*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(dim*2)
        self.deconv4 = nn.ConvTranspose2d(dim*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(dim)
        self.deconv5 = nn.ConvTranspose2d(dim, 3, 4, 2, 1)
    
    def weight_init(self, mean, std):
        for m in self._modules:
            initialize_normal(self._modules[m], mean, std)

    def forward(self, inp):
        x = F.relu(self.deconv1_bn(self.deconv1(inp)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        return x
    



