import torchvision.models as models
import torch.nn as nn

def build_model(pretrained=True, finetune=True, num_classes=10):
    if pretrained:
        print('[INFO]: Loading pretrained weights')
    else:
        print('[INFO]: Training model from scratch')
    #classification_model = models.resnext50_32x4d(pretrained=pretrained)
    classification_model = models.efficientnet_b0(pretrained=pretrained)

    if finetune:
        print('[INFO]: Fine-tuning all layers...')
        for params in classification_model.parameters():
            params.requires_grad = True
    elif not finetune:
        print('[INFO]: Freezing hidden layers')
        for params in classification_model.parameters():
            params.requires_grad = False

    # change the last classification layer with required number of classes
    #in_features = classification_model.fc.in_features
    #classification_model.fc = nn.Linear(in_features, num_classes)
    classification_model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return classification_model
