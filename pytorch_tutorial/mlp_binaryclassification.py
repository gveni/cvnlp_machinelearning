import numpy as np
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
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
        print("Input data shape:", np.shape(self.X))
        print("Input label shape:", np.shape(self.y))
        # Ensure input X values are floats
        self.X = self.X.astype('float32')
        # Encode target labels and ensure they are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape(len(self.y), 1)
        #print("After reshaping, input label shape:", np.shape(self.y))
        #print("Unique labels:", np.unique(self.y))

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
    # define model elements
    def __init__(self, n_inputs):
        super(MLP,self).__init__()
        # input to first MLP hidden layer
        self.hiddenlayer1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hiddenlayer1.weight, nonlinearity='relu')
        self.activation1 = ReLU()

        # input to second MLP hidden layer
        self.hiddenlayer2 = Linear(10, 8)
        kaiming_uniform_(self.hiddenlayer2.weight, nonlinearity='relu')
        self.activation2 = ReLU()

        # input tp third MLP hidden layer (last layer)
        self.hiddenlayer3 = Linear(8, 1)
        xavier_uniform_(self.hiddenlayer3.weight)
        self.activation3 = Sigmoid()

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
        X = self.activation3(X)
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
    loss_fn = BCELoss()
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
        # round yhats to class values
        yhats = yhats.round()
        predictions.append(yhats)
        actuals.append(ys)
   
   # stack arrays vertically that are stored in batches
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate performance metric
    acc = accuracy_score(predictions, actuals)
    return acc


# predict class for a given input
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # predict input
    yhat = model(row)
    # retrieve numpy array by detaching from pytorch tensor
    yhat = yhat.detach().numpy()
    # round the value to nearest class
    return yhat


# Driver functions
# prepare data
dataset_path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
train_dl, test_dl = prepare_data(dataset_path)
print("Number of samples in train and test set:", len(train_dl.dataset), len(test_dl.dataset))
# define network architecture
binaryclass_model = MLP(34)
# train the model
train_model(train_dl, binaryclass_model)
# evaluate model
accuracy = evaluate_model(test_dl, binaryclass_model)
print("Accuracy: %3f"%accuracy)
# predict class for a given input
row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
yhat = predict(row, binaryclass_model)
print("Predicted: %.3f and assigned class=%d"%(yhat, yhat.round()))
