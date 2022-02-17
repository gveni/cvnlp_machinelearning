"""
Fine-tune pre-trained Mask R-CNN-based object detection model
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.optim import SGD, lr_scheduler
import transforms as T
import utils
from engine import train_one_epoch, evaluate

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# Defining class for PennFudan dataset dataset
class obj_det_dataset(Dataset):
    def __init__(self, root, transforms):
            self.root = root
            self.transforms = transforms
            # load all image files, sorting them to
            # ensure that they are aligned
            self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
            self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


"""
Finetune pre-trained fasterrcnn-resnet50 model pretrained on COCO dataset
In order to also compute instance segmentation masks, mask-RCNN needs to be used
"""
def get_instance_segmentation_model(num_classes):
    # load mask-rcnn-based instance segmentation model pretrained on COCO dataset
    mask_rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # Get input features from mask-rcnn model for the classifier
    in_features = mask_rcnn_model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained model head with num_classes
    mask_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get input features for mask classifier
    mask_in_features = mask_rcnn_model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one
    mask_rcnn_model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in_features, hidden_layer, num_classes)
    return mask_rcnn_model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# Test code to see what faster-rcnn model expects during the training and evaluation of sample data
#faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#parent_dir = "/Users/gveni/Documents/Projects/UGC_PhotoMagic/Data/PennFudanPed"
#pennfudan_ds = obj_det_dataset(parent_dir, get_transform(train=True))
#pennfudan_dl = torch.utils.data.DataLoader(
#    pennfudan_ds, batch_size=2, shuffle=True, num_workers=4,
#    collate_fn=utils.collate_fn
#)
## For training
#images, targets = next(iter(pennfudan_dl))
#images = list(image for image in images)
#targets = [{key: val for key, val in target.items()} for target in targets]
#output = faster_rcnn_model(images, targets)  # return losses and detections
## For inference
#faster_rcnn_model.eval()
#x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
#predictions = faster_rcnn_model(x)           # Returns predictions


def main():
    # train on GPU if avaiable, otherwise on CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # PennFudan dataset deals with humans so only two classes: background and humans
    num_classes = 2
    parent_dir = "/home/ec2-user/ebs_xvdg/data/UGC_PhotoMagic/Data/PennFudanPed"
    dataset = obj_det_dataset(parent_dir, get_transform(train=True))
    dataset_test = obj_det_dataset(parent_dir, get_transform(train=False))
    #dataset.__get_item__(50)

    # split the data into training and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # Prepare dataloaders for training and test
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    instance_segmentation_model = get_instance_segmentation_model(num_classes)
    instance_segmentation_model.to(device)  # move model to CPU/GPU

    # Set up optimizer
    params = [p for p in instance_segmentation_model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                       step_size=3,
                                       gamma=0.1)
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(
            instance_segmentation_model,
            optimizer,
            dataloader,
            device, epoch,
            print_freq=10)
        # update learnign rate
        lr_scheduler.step()
        # evaluate on the test set
        evaluate(instance_segmentation_model,
                 dataloader_test,
                 device=device)
    print("Instance segmentation training and tst finished!!")




if __name__ == '__main__':
    main()
