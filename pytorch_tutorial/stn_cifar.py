import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import copy
from tqdm import tqdm
import time


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        # simple ConvNet classifier
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # ip size: 3x32x32 o/p size: 6x28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # o/p size: 16x10x10
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)  # o/p size: 6x14x14
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # spatial transformer localization network
        self.stn_localization = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7),  # ip size: 3x32x32 o/p size: 64x26x26
            nn.MaxPool2d(2, stride=2),  # o/p size: 64x13x13
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5),  # o/p size: 128x9x9
            nn.MaxPool2d(2, stride=2), # o/p size: 128x4x4
            nn.ReLU(True)
        )

        # spatial transformer regression network for theta estimation
        self.stn_regression = nn.Sequential(
            nn.Linear(128*4*4, 256),
            nn.ReLU(True),
            nn.Linear(256, 6)
        )

        # initialization of STN transformation weights and biases with identity transformations
        self.stn_regression[2].weight.data.zero_()
        self.stn_regression[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))

    # Combine localization and regression networks to build STN
    def stn(self, x):
        x_stn = self.stn_localization(x)
        x_stn = x_stn.view(-1, x_stn.size(1)*x_stn.size(2)*x_stn.size(3))
        # calculate theta parameters
        theta = self.stn_regression(x_stn)
        # reshape theta
        theta = theta.view(-1, 2, 3)
        # grid generator -> generate o/p image's parametrized sampling grid from i/p image using theta
        grid = F.affine_grid(theta, x.size())
        # interpolate parametrized sampling grid to produce values corresponding to o/p grid
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform input through STN network
        x = self.stn(x)
        # then pass it standard ConvNet classifier
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

# define learning parameters
data_path = '~/.torch/datasets/cifar'
workers = 1
lr = 0.001
batch_size = 64
epochs = 50
model_dir = '/home/ec2-user/Code/cvnlp_machinelearning/pytorch_tutorial/models'
os.makedirs(model_dir, exist_ok=True)
model_file = 'stn_cifar.pth'

# image transforms
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

image_transforms = {'train': transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
    ]), 'val': transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
    ])
}

# specify the computing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets = {x: datasets.CIFAR10(data_path,
                                download=True,
                                transform=image_transforms[x])
            for x in ['train', 'val']}
print("chosen datasets", datasets)

dataloaders = {
    x: DataLoader(datasets[x],
                  batch_size=batch_size,
                  shuffle=True,
                  num_workers=workers
                 )
    for x in ['train', 'val']
}

# get dataset sizes and class names for train and validation datasets
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
class_names = datasets['train'].classes

# initialize the model, optimizer, loss_function
stn_model = STN().to(device)
optimizer = optim.SGD(stn_model.parameters(), lr=lr)
loss_criteria = nn.CrossEntropyLoss()

# train STN model
def train_model(stn_model, optimizer, loss_criteria, num_epochs=epochs):
    st = time.time()
    # set initial model weights through state dictionary
    best_model_weights = copy.deepcopy(stn_model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch, num_epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                stn_model.train()
            else:
                stn_model.eval()

            current_loss = 0.0
            current_acc = 0.0
            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()  # void gradients

                with torch.set_grad_enabled(phase=='train'):
                    yhats = stn_model(inputs)
                    _, preds = torch.max(yhats.data, axis=1)
                    loss = loss_criteria(yhats, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                current_loss += loss.item() * inputs.size(0)
                current_acc += torch.sum(preds == targets.data)

            epoch_loss = current_loss / dataset_sizes[phase]
            epoch_acc = current_acc.double()/dataset_sizes[phase]

            print('{} loss: {:.4f}. accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(stn_model.state_dict())

        print()

    total_time = time.time() - st
    print('Training complete in {:.0f}minu {:.0f}sec'.format(total_time // 60, total_time % 60))
    print('Best validation accuracy {:3f}'.format(best_acc))

    # now load the best model weights for saving the model
    stn_model.load_state_dict(best_model_weights)
    torch.save(stn_model, os.path.join(model_dir, model_file))
    return stn_model


def tensor2numpy(inp):
    inp = inp.numpy().transform((1,2,0))
    inp = std*inp + mean
    return inp


def visualize_stn():
    stn_model = torch.load(os.path.join(model_dir, model_file))
    with torch.no_grad():
        data = next(iter(dataloaders['val']))[0].to(device)

        input_tensor = data.cpu()
        transformed_tensor = stn_model.stn(data).cpu()

        ip_grid = tensor2numpy(torchvision.utils.make_grid(input_tensor))
        op_grid = tensor2numpy(torchvision.utils.make_grid(transformed_tensor))

        # plot side-by-side results for comparison
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(ip_grid)
        axarr[0].set_title('input images')
        
        axarr[1].imshow(op_grid)
        axarr[1].set_title('transformed images')



trained_stn_model = train_model(stn_model, optimizer, loss_criteria, epochs)
#visualize_stn()
#plt.ioff()
#plt.show()
