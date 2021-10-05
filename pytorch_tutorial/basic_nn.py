import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Get device for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Working on {} device'.format(device))


# Define our neural network subclassing nn.Module
class BasicNN(nn.Module):
    def __init__(self):
        super(BasicNN, self).__init__()
        self.flatten = nn.Flatten()
        self.seq_linear_relu = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.seq_linear_relu(x)
        return x


# create an instance of our network, pass it to the device and print its architecture
basicnn_model = BasicNN().to(device)
print(basicnn_model)


ip_data = torch.rand(1, 28, 28, device=device)  # create a dummy input data
logits = basicnn_model(ip_data)
preds = nn.Softmax(dim=1)(logits)
y_preds = preds.argmax(dim=1)
print(f'Predicted class: {y_preds}')

