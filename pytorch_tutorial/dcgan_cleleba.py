import argparse
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch

import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# set random seed for reproducibility
randomseed = 42
random.seed(randomseed)
torch.manual_seed(randomseed)

# define inputs for DCGAN model
dataroot = "/Users/gveni/Documents/Projects/UGC_PhotoMagic/Miscelleanous_Data/celeba"  # root data dir
workers = 2  # number of workers for dataloader
batch_size = 128  # training batch size
image_size = 64  # default image size for training images
num_channels = 3
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

