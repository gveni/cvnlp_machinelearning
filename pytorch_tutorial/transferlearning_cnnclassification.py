import os
import shutil
import re

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import copy

parent_dir = '/Users/gveni/Documents/Projects/UGC_PhotoMagic/Data/dogs-vs-cats/'

files = os.listdir(parent_dir + 'train/')

# move class-based images to the respective folders
def train_maker(name):
    train_dir = parent_dir+'train/'+name
    os.makedirs(train_dir, exist_ok=True)
    for f in files:
        search_obj = re.search(name, f)
        if search_obj:
            shutil.move(os.path.join(parent_dir, 'train', f), train_dir)

#train_maker('cat')
#train_maker('dog')

# try:
#     os.makedirs(parent_dir+"val/cat")
#     os.makedirs(parent_dir+"val/dog")
# except OSError:
#     print("Creating directory failed")
# else:
#     print("Directory successfully created")

cat_train = parent_dir + "train/cat"
cat_val = parent_dir + "val/cat"
dog_train = parent_dir + "train/dog"
dog_val = parent_dir + "val/dog"

cat_files = os.listdir(cat_train)
dog_files = os.listdir(dog_train)

# Put 1000 images from class-specific training folders to the respective validation folders
# for f in cat_files:
#     validCatSearchObj = re.search("5\d\d\d", f)
#     if validCatSearchObj:
#         shutil.move(f'{cat_train}/{f}', cat_val)
#
# for f in dog_files:
#     validDogSearchObj = re.search("5\d\d\d", f)
#     if validDogSearchObj:
#         shutil.move(f'{dog_train}/{f}', dog_val)


# data augmentation using transforms
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

image_transforms = {'train': transforms.Compose([
    transforms.RandomResizedCrop(size=256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
]), 'val': transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
]),
}

chosen_datasets = {x: datasets.ImageFolder(os.path.join(parent_dir, x), image_transforms[x])
                   for x in ['train', 'val']}

# prepare dataloaders for training and validation
dataloaders = {x: DataLoader(chosen_datasets[x], batch_size=4, shuffle=4, num_workers=4)
               for x in ['train', 'val']}

# Get the dataset sizes and class names for train and validation datasets
dataset_sizes = {x: len(chosen_datasets[x]) for x in ['train', 'val']}
class_names = chosen_datasets['train'].classes

# Specify to use "GPU" if available else 'CPU'
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# visualize some images
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([mean_nums])
    std = np.array([std_nums])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(.5)  # Pause a bit so that plots are updated

# Grab some of the training data to visualize
inputs, classes = next(iter(dataloaders['train']))

# Now we construct a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

##########################
# Setting up pre-trained model
##########################

resnet_model = models.resnet18(pretrained=True)
num_features = resnet_model.fc.in_features
resnet_model.fc = torch.nn.Linear(num_features, 2)

#for name, child in resnet_model.named_children():
#    print(name)

# define loss function, optimizer and learning rate scheduler to prevent from over-shooting/non-convergence
resnet_model = resnet_model.to(device)
loss_criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(resnet_model.parameters(), lr=0.001)
# decay lr by a factor of 0.1 after every 7 epochs
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# train the model
def train_model(model, loss_criterion, optimizer, scheduler, num_epochs=10):
    st = time.time()
    # set initial models weights from the pretrained model through state_dict
    best_model_wgts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

        current_loss = 0.0
        current_acc = 0.0

        for inputs, targets in dataloaders[phase]:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # void gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                yhats = model(inputs)
                predictions = np.argmax(yhats, axis=1)
                loss = loss_criterion(yhats, targets)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            epoch_loss = current_loss / dataset_sizes[phase]
            epoch_acc = current_acc.double() / dataset_sizes[phase]

            print('{} Loss: {:.3f}. Accuracy: {:.3f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wgts = copy.deepcopy(model.state_dict())
        print()

    total_time = time.time() - st
    print('Training complete in {:.0f}m {:.0f}s'.format(
        total_time // 60, total_time % 60))
    print('Best val Acc: {:3f}'.format(best_acc))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wgts)
    return model