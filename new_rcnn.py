import torch
from torch import nn

from references import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign

from torchvision.models.utils import load_state_dict_from_url

from faster_rcnn import FasterRCNN
from backbone_utils import resnet_fpn_backbone

from collections import OrderedDict

__all__ = [
    "NewRCNN", "newrcnn_resnet50_fpn"
]

class NewRCNN(FasterRCNN):
    def __init__(self, backbone, num_classes=None,
         # transform parameters
         min_size=None, max_size=1333,
         image_mean=None, image_std=None,
         # RPN parameters
         rpn_anchor_generator=None, rpn_head=None,
         rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
         rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
         rpn_nms_thresh=0.7,
         rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
         rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
         # Box parameters
         box_roi_pool=None, box_head=None, box_predictor=None,
         box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
         box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
         box_batch_size_per_image=512, box_positive_fraction=0.25,
         bbox_reg_weights=None, roi_pool=None, 
         # Mask parameters
         mask_roi_pool=None, mask_head=None, mask_predictor=None,
         # Keypoint parameters
         keypoint_roi_pool=None, keypoint_head=None, keypoint_predictor=None,
         num_keypoints=17):

        assert isinstance(keypoint_roi_pool, (MultiScaleRoIAlign, type(None)))
        assert isinstance(mask_roi_pool, (MultiScaleRoIAlign, type(None)))        
        
        if min_size is None:
            min_size = (640, 672, 704, 736, 768, 800)        

        if num_classes is not None:
            if keypoint_predictor is not None:
                raise ValueError("num_classes should be None when keypoint_predictor is specified")
            if mask_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")
                
        out_channels = backbone.out_channels
        
        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                # output_size=(22, 16),
                output_size=14,
                sampling_ratio=2)

        if mask_head is None:
            # mask_layers = (256, 256, 256, 256)
            mask_layers = tuple(512 for _ in range(4))
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 512  # == mask_layers[-1]
            mask_dim_reduced = 512
            mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                               mask_dim_reduced, num_classes)
        
        if keypoint_roi_pool is None:
            keypoint_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                # output_size=(22, 16),
                output_size=14,
                sampling_ratio=2)
        
        if keypoint_head is None:
            keypoint_layers = tuple(512 for _ in range(4))
            # keypoint_head = KeypointRCNNHeads(out_channels, keypoint_layers)
            keypoint_head = KeypointRCNNHeads(512, keypoint_layers)

        if keypoint_predictor is None:
            keypoint_dim_reduced = 512  # == keypoint_layers[-1]
            keypoint_predictor = KeypointRCNNPredictor(keypoint_dim_reduced, num_keypoints)
            
        super(NewRCNN, self).__init__(
            backbone, num_classes,
            # transform parameters
            min_size, max_size,
            image_mean, image_std,
            # RPN-specific parameters
            rpn_anchor_generator, rpn_head,
            rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            # Box parameters
            box_roi_pool, box_head, box_predictor,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights)
        
        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor
        
        self.roi_heads.keypoint_roi_pool = keypoint_roi_pool
        self.roi_heads.keypoint_head = keypoint_head
        self.roi_heads.keypoint_predictor = keypoint_predictor
        
class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Arguments:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d["mask_fcn{}".format(layer_idx)] = misc_nn_ops.Conv2d(
                next_feature, layer_features, kernel_size=3,
                stride=1, padding=dilation, dilation=dilation)
            d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super(MaskRCNNHeads, self).__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNPredictor, self).__init__(OrderedDict([
            ("conv5_mask", misc_nn_ops.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("mask_fcn_logits", misc_nn_ops.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
        ]))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)

class KeypointRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers):
        d = []
        next_feature = in_channels
        for l in layers:
            d.append(misc_nn_ops.Conv2d(next_feature, l, 3, stride=1, padding=1))
            d.append(nn.ReLU(inplace=True))
            next_feature = l
        super(KeypointRCNNHeads, self).__init__(*d)
        for m in self.children():
            if isinstance(m, misc_nn_ops.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

class KeypointRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super(KeypointRCNNPredictor, self).__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = misc_nn_ops.ConvTranspose2d(
            input_features,
            input_features,
            # num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        self.relu = nn.ReLU(inplace=True)
        self.kps_score = misc_nn_ops.Conv2d(input_features, num_keypoints, 1, 1, 0)
        nn.init.kaiming_normal_(
            self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        x = self.relu(x)
        x = self.kps_score(x)
        x = misc_nn_ops.interpolate(
            x, scale_factor=(float(self.up_scale), float(self.up_scale)), mode="bilinear", align_corners=False
        )
        return x

model_urls = {
    ## TODO : make link
    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
    'keypointrcnn_resnet50_fpn_coco_legacy':
        'https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-9f466800.pth',
    'keypointrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth',
}        


def newrcnn_resnet50_fpn(pretrained=False, progress=True, weights=None,
                         num_classes=2, num_keypoints=17, 
                         pretrained_backbone=True, **kwargs):
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    model = NewRCNN(backbone, num_classes, **kwargs)
    if pretrained:
# state_dict = load_state_dict_from_url(model_urls['maskrcnn_resnet50_fpn_coco'],
#                                              progress=progress)
        #weight_path = "new_rcnn_resnet50_fpn.pth"
        weight_path = "weight/newrcnn.pth"
        if weights is not None :
            weight_path = "weight/" + weights
        state_dict = torch.load(weight_path)
        model.load_state_dict(state_dict)
    return model
