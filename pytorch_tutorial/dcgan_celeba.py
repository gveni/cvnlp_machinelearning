import argparse
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch

import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn

# set random seed for reproducibility
randomseed = 42
random.seed(randomseed)
torch.manual_seed(randomseed)

# define inputs for DCGAN model
dataroot = "/home/ec2-user/ebs_xvdg/data/UGC_PhotoMagic/Miscelleanous_Data/celeba"  # root data dir
workers = 2  # number of workers for dataloader
batch_size = 128  # training batch size
image_size = 64  # default image size for training images
nc = 3  # number of channels (3: RGB)
nz = 100  # length of latent vector, z
ngf = 64  # depth of feature maps carried through the generator
ndf = 64  # depth of feature maps carried through the discriminator
num_epochs = 5
lr = 0.0002
beta1 = 0.5  # initial decay rate of ADAM optimizer for first moment of gradient
ngpu = 1


# create a dataset  and dataloader class
dataset = dataset.ImageFolder(root=dataroot,
                              transform=transforms.Compose([
                                  transforms.Resize(image_size),
                                  transforms.CenterCrop(image_size),
                                  transforms.
                                  ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])
                              )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# plot some samples
batch = next(iter(dataloader))
#plt.figure(figsize=(8,8))
#plt.axis('off')
#plt.title("Data samples")
#plt.imshow(np.transpose(vutils.make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# setting up generator architecture
class Generator(nn.Module):
    def __init__(self, ngpus):
        super(Generator, self).__init__()
        self.ngpus = ngpus
        self.main = nn.Sequential(
            # first convolution-transpose, BatchNorm, ReLU layers. input: (nz). output: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # second convolution-transpose, BatchNorm, ReLU layers. input: (ngf*8) x 4 x 4. output: (ngf*4) * 8 x 8
            nn.ConvTranspose2d(in_channels=ngf*8, out_channels=ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # third convolution-transpose, BatchNorm, ReLU layers. input: (ngf*4) x 8 x 8. output: (ngf*2) * 16 x 16
            nn.ConvTranspose2d(in_channels=ngf*4, out_channels=ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(False),
            # fourth convolution-transpose, BatchNorm, ReLU layers. input: (ngf*2) x 16 x 16. output: (ngf) * 32 x 32
            nn.ConvTranspose2d(in_channels=ngf*2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(False),
            # fifth convolution-transpose, BatchNorm, ReLU layers. input: (ngf) x 32 x 32. output: (nc) * 64 x 64
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)

# Instantiate the generator and re-initialize the weights
netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(netG)  # print generator model architecture

# setting up discriminator architecture
class Discriminator(nn.Module):
    def __init__(self, ngpus):
        super(Discriminator, self).__init__()
        self.ngpus = ngpus
        self.main = nn.Sequential(
            # first convolution, Leaky-ReLU layers. input: (nc) x 64 x 64, output: (ngf) x 32 x 32
            nn.Conv2d(in_channels=nc, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # second convolution, batch-norm, leakyReLU layers. input: (ngf) x 32 x 32, output: (ngf*2) x 16 x 16
            nn.Conv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # third convolution, batch-norm, leakyReLU layers. input: (ngf*2) x 16 x 16, output: (ngf*4) x 8 x 8
            nn.Conv2d(in_channels=ngf*2, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # fourth convolution, batch-norm, leakyReLU layers. input: (ngf*4) x 8 x 8, output: (ngf*8) x 4 x 4
            nn.Conv2d(in_channels=ngf*4, out_channels=ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # fifth convolution, BatchNorm, Sigmoid (for 2-classes) layers. input: (ngf*8) x 4 x 4, output: 1
            nn.Conv2d(in_channels=ngf*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)

# Instantiate the discriminator and re-initialize the weights
netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
print(netD)  # print discriminator model architecture
