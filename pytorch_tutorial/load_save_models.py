import torch
import torch.onnx as onnx
import torchvision.models as models


"""
Save and load model weights
"""
# save model weights
model2save = models.vgg16(pretrained=True)
torch.save(model2save.state_dict(), "model_weights.pth")

# load model weights
model2load = models.vgg16()  # do not specify pretrained=True as it loads default weights
model2load.load_state_dict(torch.load("model_weights.pth"))
model2load.eval()  # this will set dropout and batch normalization layers to evaluation mode


"""
Save and load models with shape
"""
torch.save(model2save, "model.pth")
model2load = torch.load("model.pth")
