'''
Visualize pre-trained object detector on
GT Dataset
'''
from dataloaders.mscoco import CocoDetection, CocoDataLoader
from dataloaders.georgia_tech import GTDataLoader, GT
from lib.object_detector import ObjectDetector
import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os
from config import ModelConfig, FG_FRACTION, RPN_FG_FRACTION, IM_SCALE_GT, BOX_SCALE_GT
from torch.nn import functional as F
from lib.fpn.box_utils import bbox_loss
import torch.backends.cudnn as cudnn
from pycocotools.cocoeval import COCOeval
from lib.pytorch_misc import optimistic_restore, clip_grad_norm
from torch.optim.lr_scheduler import ReduceLROnPlateau

cudnn.benchmark = True
conf = ModelConfig()

# load testing dataset
train, val, test = GT.splits(num_val_im=conf.val_size, filter_non_overlap=False,
                          filter_empty_rels=False, use_proposals=conf.use_proposals)
# val = test
train_loader, val_loader = GTDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

# Initialize the model
detector = ObjectDetector(classes=train.ind_to_classes, num_gpus=conf.num_gpus,
                          mode='rpntrain' if not conf.use_proposals else 'proposals', use_resnet=conf.use_resnet)
# Load pre-trained weights
ckpt = torch.load(conf.ckpt)
optimistic_restore(detector, ckpt['state_dict'])

# Append model to Cuda device
detector.cuda()

# Start evaluating
print('Start evaluating')
def val_epoch():
    detector.eval()
    # all_boxes is a list of length number-of-classes.
    # Each list element is a list of length number-of-images.
    # Each of those list elements is either an empty list []
    # or a numpy array of detection.
    vr = []
    for val_b, batch in enumerate(val_loader):
        vr.append(val_batch(val_b, batch))
    vr = np.concatenate(vr, 0)
    if vr.shape[0] == 0:
        print("No detections anywhere")
        return 0.0

    val_coco = val.coco
    coco_dt = val_coco.loadRes(vr)
    coco_eval = COCOeval(val_coco, coco_dt, 'bbox')
    coco_eval.params.imgIds = val.ids if conf.coco else [x for x in range(len(val))]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAp = coco_eval.stats[1]
    return mAp


def val_batch(batch_num, b):
    result = detector[b]
    if result is None:
        return np.zeros((0, 7))
    scores_np = result.obj_scores.data.cpu().numpy()
    cls_preds_np = result.obj_preds.data.cpu().numpy()
    boxes_np = result.boxes_assigned.data.cpu().numpy()
    im_inds_np = result.im_inds.data.cpu().numpy()
    im_scales = b.im_sizes.reshape((-1, 3))[:, 2]
    if conf.coco:
        boxes_np /= im_scales[im_inds_np][:, None]
        boxes_np[:, 2:4] = boxes_np[:, 2:4] - boxes_np[:, 0:2] + 1
        cls_preds_np[:] = [val.ind_to_id[c_ind] for c_ind in cls_preds_np]
        im_inds_np[:] = [val.ids[im_ind + batch_num * conf.batch_size * conf.num_gpus]
                         for im_ind in im_inds_np]
    else:
        boxes_np *= BOX_SCALE_GT / IM_SCALE_GT
        boxes_np[:, 2:4] = boxes_np[:, 2:4] - boxes_np[:, 0:2] + 1
        im_inds_np += batch_num * conf.batch_size * conf.num_gpus

    return np.column_stack((im_inds_np, boxes_np, scores_np, cls_preds_np))

mAp = val_epoch()
print('mAP on val dataset: {}'.format(mAp))