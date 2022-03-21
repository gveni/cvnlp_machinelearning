# This file contains helper functions to save the trained model and outputs
import torch
import matplotlib.pyplot as plt

# Save the current status of the model including the number of epochs, the model at its current stage, the optimizer,
# and the loss function. This will also facilitate to resume training later if needed
def save_model(num_epochs, model, optimizer, loss_fn, pretrained):
    """
    Function to save the trained model
    :param num_epochs: number of epochs
    :param model: model to be saved at its current stage
    :param optimizer: chosen optimizer
    :param loss_fn: loss function used
    :param pretrained: boolean value to indicate if pretrained model is used
    :return: Null
    """
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_fn,
    }, f"../outputs/model_pretrained_{pretrained}.pth")


# Save output graphs
def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained):
    """
    function to save training and validation accuracy and loss over epochs
    :param train_acc:
    :param valid_acc:
    :param train_loss:
    :param valid_loss:
    :param pretrained:
    :return: Null
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color='green', linestyle='-', label='train accuracy')
    plt.plot(valid_acc, color='blue', linestyle='-', label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(f"../outputs/accuracy_pretrained_{pretrained}.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', linestyle='-', label='train loss')
    plt.plot(valid_loss, color='red', linestyle='-', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f"../outputs/loss_pretrained_{pretrained}.png")
