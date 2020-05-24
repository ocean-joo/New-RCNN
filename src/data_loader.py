import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO

class NewRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, root="/mnt/ssd/Data/COCO/train2017", 
            annotation="/mnt/ssd/Data/COCO/annotations/person_keypoints_train2017.json", 
            transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path))

        num_objs = len(coco_annotation)

        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        img_id = torch.tensor([img_id])
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        if 1 not in labels and num_objs is 0 :
            keypoints = torch.as_tensor([])
        else :
            keypoints = torch.empty(num_objs, 17, 3)        
        for i in range(num_objs):
            if len(coco_annotation[i]['keypoints']) != 0:
                keypoint = torch.as_tensor(coco_annotation[i]['keypoints']).reshape(17, 3)
                keypoints[i] = keypoint
        # masks
        masks = []
        for i in range(num_objs) :
            masks.append(coco.annToMask(coco_annotation[i]))
        masks = torch.as_tensor(masks)

        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        my_annotation["keypoints"] = keypoints
        my_annotation["masks"] = masks

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)
