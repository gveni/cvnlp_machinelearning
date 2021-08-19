from numpy import argmax, vstack
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.nn import Module, Conv2d, MaxPool2d, Linear, ReLU, Softmax, CrossEntropyLoss
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.optim import Adam
from sklearn.metrics import accuracy_score


def prepare_data(path):
    # define transforms to be applied to the dataset
    transforms = Compose([ToTensor()])
    # define the dataset
    train_ds = MNIST(path, train=True, download=True, transform=transforms)
    test_ds = MNIST(path, train=False, download=True, transform=transforms)
    # load datasets
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
    return train_dl, test_dl


class CNN(Module):
    def __init__(self, num_channels):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(num_channels, 32, (3, 3))
        kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        self.activation1 = ReLU()
        self.pool1 = MaxPool2d((2, 2), stride=(2, 2))
        self.conv2 = Conv2d(32, 32, (3, 3))
        kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        self.activation2 = ReLU()
        self.pool2 = MaxPool2d((2,2), stride=(2,2))
        self.fcl1 = Linear(5*5*32, 100)
        kaiming_uniform_(self.fcl1.weight, nonlinearity='relu')
        self.activation3 = ReLU()
        self.fcl2 = Linear(100, 10)
        xavier_uniform_(self.fcl2.weight)
        self.activation4 = Softmax(dim=1)

    def forward(self, X):
        X = self.conv1(X)
        X = self.activation1(X)
        X = self.pool1(X)
        X = self.conv2(X)
        X = self.activation2(X)
        X = self.pool2(X)
        # Flatten
        X = X.view(-1, 4*4*50)
        X = self.fcl1(X)
        X = self.activation3(X)
        X = self.fcl2(X)
        X = self.activation4(X)
        return X


def train_model(train_dl, model, learning_rate, num_epochs):
    lossfn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhats = model(inputs)
            loss = lossfn(yhats, targets)
            loss.backward()
            optimizer.step()


def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        yhat = argmax(yhat, axis=1)
        yhat = yhat.reshape((len(yhat), 1))
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    accuracy = accuracy_score(predictions, actuals)
    return accuracy


path = '~/.torch/datasets/mnist'
train_dl, test_dl = prepare_data(path)
print("Train and test data sizes", len(train_dl.dataset), len(test_dl.dataset))
num_channels = 1
cnn_model = CNN(num_channels)
learning_rate = 1e-5
num_epochs = 10
train_model(train_dl, cnn_model, learning_rate, num_epochs)
accuracy = evaluate_model(test_dl, cnn_model)