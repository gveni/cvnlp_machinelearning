import numpy as np
from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


# model parameters
num_epochs = 100

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load csv file as pandas dataframe
        df = read_csv(path, header=None)
        # extracts inputs and outputs from dataframe
        self.X = df.values[:,:-1]
        self.y = df.values[:,-1]
        print("Input data shape", np.shape(self.X))
        print("Input label shape", np.shape(self.y))
        # ensure inputs as floats
        self.X = self.X.astype('float32')
        # encode target values and ensure they are floats
        self.y = LabelEncoder().fit_transform(self.y)

    # Number of samples in dataset
    def __len__(self):
        return(len(self.X))

    # Get row/data at specific index
    def __getitem__(self, idx):
        return([self.X[idx], self.y[idx]])

    # Get indices for train and test data
    def get_splits(self, n_train=0.67):
        # determine train and test split sizes
        train_split = round(n_train * len(self.X))
        test_split = len(self.X) - train_split
        # randomly split data into train and test
        return random_split(self, [train_split, test_split])


# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.activation1 = ReLU()
        # input to second hidden layer
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.activation2 = ReLU()
        # input to third hidden layer
        self.hidden3 = Linear(8, 3)
        xavier_uniform_(self.hidden3.weight)
        self.activation3 = Softmax(dim=1)

    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.activation1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.activation2(X)
        X = self.hidden3(X)
        X = self.activation3(X)
        return(X)

# prepare data
def prepare_data(path):
    iris_dataset = CSVDataset(path)
    # split data into train and test
    train, test = iris_dataset.get_splits()
    # prepare train and test data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=32, shuffle=False)
    return train_dl, test_dl


# train the model
def train_model(train_dl, model):
    # define loss function
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # enumerate the model
    for epoch in range(num_epochs):
        # enumerate through batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear gradients
            optimizer.zero_grad()
            # compute model output
            yhats = model(inputs)
            # calculate loss
            loss = loss_fn(yhats, targets)
            # back propagate
            loss.backward()
            # update model weights
            optimizer.step()


# evaluate model
def evaluate_model(test_dl, model):
    actuals, predictions = list(), list()
    # enumerate through test batches
    for i, (inputs, targets) in enumerate(test_dl):
        # predict test outputs
        yhats = model(inputs)
        # convert to class labels by first converting pytorch tensors to numpy
        yhats = yhats.detach().numpy()
        yhats = argmax(yhats, axis=1)
        ys = targets.numpy()
        # reshape for stacking
        yhats= yhats.reshape(len(yhats), 1)
        ys = ys.reshape(len(ys), 1)
        # store
        actuals.append(ys)
        predictions.append(yhats)

    actuals, predictions = vstack(actuals), vstack(predictions)
    accuracy = accuracy_score(actuals, predictions)
    return accuracy


# test model on a single sample
def test_model(row, model):
    # convert row to pytorch tensor
    input = Tensor([row])
    yhat = model(input)  # predict output
    yhat = yhat.detach().numpy()
    return yhat


# driver/main code
# prepare data
dataset_path  = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
train_dl, test_dl = prepare_data(dataset_path)
print('Number of samples in train and test datasets:', len(train_dl.dataset), len(test_dl.dataset))
# define the model
multiclass_model = MLP(4)
# train the model
train_model(train_dl, multiclass_model)
# evaluate model
accuracy = evaluate_model(test_dl, multiclass_model)
print("Accuracy of multiclass MLP model on iris dataset is %.3f"%accuracy)
# test model on a single sample
row = [5.1,3.5,1.4,0.2]
yhat = test_model(row, multiclass_model)
print("Predicted class probabilities = %s and assigned class = %d" %(yhat, argmax(yhat)))
