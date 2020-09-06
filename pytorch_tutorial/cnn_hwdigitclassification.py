from numpy import vstack, argmax
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
from torch.nn import Module
from torch.nn import Conv2d, ReLU, MaxPool2d
from torch.nn import Linear
from torch.nn import Softmax
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.optim import SGD


# model parameters
num_epochs = 10
learning_rate = 0.01


# model definition
class CNN(Module):
    # define model architecture
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        # input to first hidden layer
        self.convlayer1 = Conv2d(n_channels, 32, (3,3))
        kaiming_uniform_(self.convlayer1.weight, nonlinearity='relu')
        self.actlayer1 = ReLU()
        self.pool1 = MaxPool2d((2,2), stride=(2,2))  # first pooling layer
        # input to second hidden layer
        self.convlayer2 = Conv2d(32, 32, (3,3))
        kaiming_uniform_(self.convlayer2.weight, nonlinearity='relu')
        self.actlayer2 = ReLU()
        self.pool2 = MaxPool2d((2,2), stride=(2,2)) # second pooling layer
        # first fully connected layer
        self.fcnlayer1 = Linear(5*5*32, 100)
        kaiming_uniform_(self.fcnlayer1.weight, nonlinearity='relu')
        self.actlayer3 = ReLU()
        # output layer
        self.fcnlayer2 = Linear(100, 10)
        xavier_uniform_(self.fcnlayer2.weight)
        self.actlayer4 = Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        # input to first convolution layer
        X = self.convlayer1(X)
        X = self.actlayer1(X)
        X = self.pool1(X)
        # input to second convolution layer
        X = self.convlayer2(X)
        X = self.actlayer2(X)
        X = self.pool2(X)
        # flatten 
        X = X.view(-1, 5*5*32)
        # input to first fully connected layer
        X = self.fcnlayer1(X)
        X = self.actlayer3(X)
        # output layer
        X = self.fcnlayer2(X)
        X = self.actlayer4(X)
        return X


# prepare data
def prepare_data(path):
    # define standardization
    transforms = Compose([ToTensor(), Normalize((0.1307,),(0.3081,))])
    # load MNIST dataset
    train = MNIST(path, train=True, download=True, transform=transforms)
    test = MNIST(path, train=False, download=True, transform=transforms)
    # Extract a subset of data from train dataset to reduce training time
    train_subset = Subset(train, range(50000))
    # prepare data loaders
    train_dl = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


# train CNN model
def train_model(train_dl, model):
    # define loss function and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # enumerate epochs
    for epoch in range(num_epochs):
        # enumerate over batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear previous gradients
            optimizer.zero_grad()
            # compute model output
            yhat = model(inputs)
            # calculate loss
            loss = loss_fn(yhat, targets)
            # backpropagate error
            loss.backward()
            # update model weights
            optimizer.step()


# evaluate CNN model on test/dev data
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # compute model output and take argmax based on probabilities
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        yhat = argmax(yhat, axis=1)
        # extract numpy arrays for targets
        y = targets.numpy()
        # reshape numpy arrays to 1D matrix
        yhat = yhat.reshape((len(yhat),1))
        y = y.reshape((len(y),1))
        predictions.append(yhat)
        actuals.append(y)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    accuracy = accuracy_score(predictions, actuals)
    return accuracy


# driver/main function
path = '~/.torch/datasets/mnist'
train_dl, test_dl = prepare_data(path)
# print train and test data sizes
print("Train data size", len(train_dl.dataset))
print("Test data size", len(test_dl.dataset))
# define the network
cnn_model = CNN(1)
# train CNN model
train_model(train_dl, cnn_model)
# evaluate CNN model
accuracy = evaluate_model(test_dl, cnn_model)
print("Accuracy on test set: %.3f" %accuracy)
