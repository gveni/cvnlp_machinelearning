import numpy as np
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU, Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


# Model hyper-parameters
num_epochs = 100

# Dataset definition: Data loading
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load csv file as a dataframe using pandas
        df = read_csv(path, header=None)
        # store inputs and outputs
        self.X = df.values[:,:-1]
        self.y = df.values[:,-1]
        #print("Input data shape:", np.shape(self.X))
        #print("Input label shape:", np.shape(self.y))
        # Ensure input X and y values are floats
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.y = self.y.reshape(len(self.y), 1)
        print("input data shape:", np.shape(self.X))

    # Number of rows in dataset
    def __len__(self):
        return len(self.X)

    # Get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # Get indices for train and test rows
    def get_splits(self, n_train=0.7):
        train_split = round(n_train * len(self.X))
        test_split = len(self.X) - train_split
        return random_split(self, [train_split, test_split])


# Model definition
class MLP(Module):
    # define model architecture
    def __init__(self, n_inputs):
        super(MLP,self).__init__()
        # input to first MLP hidden layer
        self.hiddenlayer1 = Linear(n_inputs, 10)
        xavier_uniform_(self.hiddenlayer1.weight)
        self.activation1 = Sigmoid()
        # input to second MLP hidden layer
        self.hiddenlayer2 = Linear(10, 8)
        xavier_uniform_(self.hiddenlayer2.weight)
        self.activation2 = Sigmoid()
        # input tp third MLP hidden layer (last layer)
        self.hiddenlayer3 = Linear(8, 1)
        xavier_uniform_(self.hiddenlayer3.weight)

    # forward propagate input
    def forward(self, X):
        # input to first MLP hidden layer
        X = self.hiddenlayer1(X)
        X = self.activation1(X)
        # input to second MLP hidden layer
        X = self.hiddenlayer2(X)
        X = self.activation2(X)
        # input to third MLP hidden layer
        X = self.hiddenlayer3(X)
        return X


# prepare dataset
def prepare_data(path):
    # Load dataset
    dataset = CSVDataset(path)
    # Calculate split
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=32, shuffle=False)
    return train_dl, test_dl


# model training
def train_model(train_dl, model):
    # define optimization
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # enumerate epochs
    for epoch in range(num_epochs):
        # enumerate mini batches
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
    predictions, actuals = list(), list()
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(test_dl):
        yhats = model(inputs)
        # retrieve numpy array from pytorch tensor
        yhats = yhats.detach().numpy()
        ys = targets.numpy()
        predictions.append(yhats)
        actuals.append(ys)
   
   # stack arrays vertically that are stored in batches
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate performance metric
    acc = mean_squared_error(predictions, actuals)
    return acc


# predict class for a given input
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # predict input
    yhat = model(row)
    # retrieve numpy array by detaching from pytorch tensor
    yhat = yhat.detach().numpy()
    return yhat


# Driver functions
# prepare data
dataset_path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
train_dl, test_dl = prepare_data(dataset_path)
print("Number of samples in train and test set:", len(train_dl.dataset), len(test_dl.dataset))
# define network architecture
mlpregression_model = MLP(13)
# train the model
train_model(train_dl, mlpregression_model)
# evaluate model
mse = evaluate_model(test_dl, mlpregression_model)
print("Mean square error: %3f. Root mean square error: %f"%(mse, sqrt(mse)))
# predict class for a given input
row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98] 
yhat = predict(row, mlpregression_model)
print("Predicted: %.3f"%yhat)
