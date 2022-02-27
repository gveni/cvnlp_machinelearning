import argparse
import os
import sys
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

# define losses, real, fake label conventions and optimizers
loss_criteria = nn.BCELoss()
real_label = 1.
fake_label = 0.
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

# Create a batch of latent vectors
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

print("Training DCGAN on celebA dataset...")
lossG_progress = []
lossD_progress = []
img_list = []

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        # Train Discriminator: maximize log(D(x)) + log(1-(D(G(z))))
        # 1. on real examples
        # clear discriminator gradients
        netD.zero_grad()
        real_batch = data[0].to(device)
        batch_sz = real_batch.size(0)
        label = torch.full((batch_sz,), real_label, dtype=torch.float, device=device)
        # forward pass real example batch through discriminator
        output = netD(real_batch)
        # calculate discriminator loss
        lossD_real = loss_criteria(output, label)
        # backpropagate loss to compute gradients
        lossD_real.backward()
        D_x = output.mean().item()

        # 2. on fake examples
        # generate a batch of latent vectors
        noise = torch.randn(batch_sz, nz, 1, 1, device=device)
        # generate fake images
        fake_batch = netG(noise)
        label.fill_(fake_label)
        # forward pass fake batch through discriminator
        output = netD(fake_batch)
        # calculate discriminator loss
        lossD_fake = loss_criteria(output, label)
        # backpropagate loss
        lossD_fake.backward()
        D_G_z1 = output.mean().item()
        # compute D by summing D_real and D_fake losses
        loss_D = lossD_real + lossD_fake
        # optimize D by updating weights
        optimizerD.step()

        # Train Generator: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since D is updated,
        output = netD(fake_batch)
        # Calculate G's loss based on this output
        loss_G = loss_criteria(output, label)
        # backpropagate loss
        loss_G.backward()
        D_G_z2 = output.mean().item()
        # optimize G by updating weights
        optimizerG.step()

        if i % 50 == 0:
            print("[%d/%d][%d/%d]\tLoss_D: %.3f\tLoss_G: %.3f\tD_x: %.3f\tD_G_z1: %.3f\t D_G_z2: %.3f)"
                  % (epoch, num_epochs, i, len(dataloader), loss_D.item(), loss_G.ite(), D_x, D_G_z1, D_G_z2))

        # save losses progress
        lossG_progress.append(loss_G.item())
        lossD_progress.append(loss_D.item())

    # Track generator output at regular intervals
    with torch.no_grad():
        fake_output = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake_output, padding=2, normalize=True))
