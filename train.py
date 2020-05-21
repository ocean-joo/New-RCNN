import os
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.tensorboard import SummaryWriter

from references.engine import *
from data_loader import NewRCNNDataset
from new_rcnn import newrcnn_resnet50_fpn
 
def collate_fn(batch) :
    return tuple(zip(*batch))
            
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('log_dir/new_rcnn_experiment')

# Dataset
transforms = torchvision.transforms.ToTensor()
# train_dataset = NewRCNNDataset(transforms=transforms)
train_dataset = NewRCNNDataset(annotation="/mnt/ssd/Data/COCO/annotations/person_keypoints_train2017.json", transforms=transforms)

print('[*] Dataset successfully loaded')

# DataLoader
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=3,
                                                shuffle=True,
                                                num_workers=1,
                                                collate_fn=collate_fn)


# Pretrained Estimator
# model = newrcnn_resnet50_fpn(pretrained=True)
model = newrcnn_resnet50_fpn(pretrained=True, weights="new_rcnn.pth")
print('[*] Model successfully loaded')

'''
# freeze branch
for name, param in model.named_parameters():
    if "keypoint" in name or "mask" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
'''

print("[*] Use #{} GPU.".format(torch.cuda.device_count()))
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.00000005, 
							momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 30

for epoch in range(9, num_epochs) :
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=1000, writer=writer)
        
    # update the learning rate
    lr_scheduler.step()
    
    torch.save(model.state_dict(), "weight/new_rcnn_modified_{}.pth".format(epoch))

writer.close()
