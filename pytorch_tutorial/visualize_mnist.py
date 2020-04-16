# load mnist dtaset and visualize it
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from matplotlib import pyplot

# define the location to save/load mnist
path = '~/.torch/datasets/mnist'
# define the transforms to apply to the data
transforms = Compose([ToTensor()])
# download and define train and test datasets
train = MNIST(path, train=True, download=True, transform=transforms)
test = MNIST(path, train=False, download=True, transform=transforms)
# load train and test datasets
train_dl = DataLoader(train, batch_size=32, shuffle=True)
test_dl = DataLoader(test, batch_size=32, shuffle=False)
# enumerate dataset and get one batch of images
i, (inputs, targets) = next(enumerate(train_dl))
for i in range(25):
    # define subplot
    pyplot.subplot(5,5,i+1)
    # plot raw pixel data
    pyplot.imshow(inputs[i][0], cmap='gray')
    pyplot.axis('off')
# show figure
pyplot.show()
