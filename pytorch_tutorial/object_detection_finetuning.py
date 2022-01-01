"""
Fine-tune pre-trained Mask R-CNN-based object detection model
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.optim import SGD, lr_scheduler
from references.detection import transforms as T
from references.detection import utils
from references.detection.engine import train_one_epoch, evaluate

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# Defining class for PennFudan dataset dataset
class obj_det_dataset(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        # sort image and mask files to ensurte they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, 'PNGImages'))))
        self.masks = list(sorted(os.listdir(os.path.join(root_dir, 'PedMasks'))))

    def __get_item__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root_dir, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "PedMasks", self.masks[idx])
        pil_img = Image.open(img_path).convert('RGB')
        pil_mask = Image.open(mask_path)
        # convert mask image to numpy array
        mask_array = np.array(pil_mask)
        # Get number of object instances (here instances are humans)
        obj_ids = np.unique(mask_array)
        # ID == 0 is background that needs to be removed
        obj_ids = obj_ids[1:]

        # split mask into instance-based individual masks
        masks = mask_array == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        bboxes = []
        for i in range(num_objs):
            bbox_pos = np.where(masks[i])
            xmin = np.min(bbox_pos[1])
            xmax = np.max(bbox_pos[1])
            ymin = np.min(bbox_pos[0])
            ymax = np.max(bbox_pos[0])
            bboxes.append([xmin, ymin, xmax, ymax])

        # Convert all target-related variables to tensors
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        # for objects without humans
        ishuman = torch.zeros((num_objs,), dtype=torch.int64)

        # encapsulate all variables into a target dictionary
        target = {}
        target['bboxes'] = bboxes  # coordinates of the N bounding boxes in [x0, y0, x1, y1]
        target['labels'] = labels  # labels ffor each boundign box
        target['masks'] = masks  # segmentation masks for object intances
        target['image_id'] = image_id  # image identifier
        target['area'] = area  # area of each bounding box
        target['ishuman'] = ishuman  # instances with ishuman=True will be ignored during evaluation.

        if self.transforms is not None:
            pil_img, target = self.transforms(pil_img, target)

        return pil_img, target


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
    parent_dir = "/Users/gveni/Documents/Projects/UGC_PhotoMagic/Data/PennFudanPed"
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

    instance_segmentation_model = get_instance_segmentation_model(num_classes=num_classes)
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