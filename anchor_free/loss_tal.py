# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import bbox_iou_v6
from anchor_free.tal.anchor_generator import dist2bbox, make_anchors, bbox2dist, distoff2bbox
from anchor_free.tal.assigner import TaskAlignedAssigner, TaskAlignedAssigner_Guided
from utils.torch_utils import de_parallel
from utils.loss_rep import repulsion_loss_torch

def xywh2xyxy_v6(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(),
                                                       reduction="none") * weight).sum()
        return loss


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # iou loss
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).view(-1, 4)
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).view(-1, 4)
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou_v6(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
        loss_iou = 1.0 - iou

        loss_iou *= bbox_weight
        loss_iou = loss_iou.sum() / target_scores_sum
        # loss_iou = loss_iou.mean()
        # print(target_bboxes_pos.size())

        # dfl loss
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).view(-1, 4, self.reg_max + 1)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).view(-1, 4)
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, iou

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none").view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1),
                                     reduction="none").view(target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

class BboxLoss_KD(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,\
                pred_dist_tea, pred_bboxes_tea):
        # iou loss
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).view(-1, 4)
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).view(-1, 4)
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou_v6(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
        loss_iou = 1.0 - iou

        loss_iou *= bbox_weight
        loss_iou = loss_iou.sum() / target_scores_sum

        

        # kd iou loss
        pred_bboxes_tea_pos = torch.masked_select(pred_bboxes_tea, bbox_mask).view(-1, 4)
        iou_KD = bbox_iou_v6(pred_bboxes_pos, pred_bboxes_tea_pos, xywh=False, CIoU=True)
        loss_iou_KD = 1.0 - iou_KD
        loss_iou_KD = loss_iou_KD.sum() / target_scores_sum
        loss_iou_KD *= 0.25
        # loss_iou = loss_iou.mean()
        # print(target_bboxes_pos.size())

        # dfl loss
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).view(-1, 4, self.reg_max + 1)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).view(-1, 4)
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum

            # KD dfl loss
            pred_dist_tea_pos = torch.masked_select(pred_dist_tea, dist_mask).view(-1, 4, self.reg_max + 1)
            # print(pred_dist_pos.size(), target_ltrb_pos.size(), pred_dist_tea_pos.size())
            pred_dist_logits = F.softmax(pred_dist_pos, dim=-1)
            pred_dist_tea_logits = F.softmax(pred_dist_tea_pos, dim=-1)
            loss_kL =  F.kl_div(pred_dist_logits.log(), pred_dist_tea_logits, reduction='batchmean').unsqueeze(0) / target_scores_sum
            loss_kL = 0.25 * loss_kL
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        # print(loss_kL.size())
        return loss_iou, loss_dfl, loss_iou_KD, loss_kL, iou

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none").view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1),
                                     reduction="none").view(target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)
    
class ComputeLoss:
    # Compute losses
    def __init__(self, model, use_dfl=True, tal_topk=10):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            print("using FocalLoss")
            BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.device = device

        self.assigner = TaskAlignedAssigner(topk=tal_topk,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                
                if n:   
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy_v6(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, p, targets, img=None):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_distri, pred_scores = p
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        # targets
        targets = self.preprocess(targets[:,:6], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = target_scores.sum()

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # if self.nc > 1:
        loss[1] = self.BCEcls(pred_scores, target_scores.to(dtype)).sum()  # BCE
        if target_scores_sum > 1:
            loss[1] /=  target_scores_sum # avoid devide zero error, devide by zero will cause loss to be inf or nan.

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2], iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    


class ComputeLoss_aux:
    # Compute losses
    def __init__(self, model, use_dfl=True, tal_topk=10):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.device = device

        self.assigner = TaskAlignedAssigner(topk=tal_topk,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        
        self.assigner_aux = TaskAlignedAssigner(topk=tal_topk,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.bbox_loss_aux = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                
                if n:   
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy_v6(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, p, targets, img=None):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_distri, pred_scores, feats_aux, pred_distri_aux, pred_scores_aux = p

        # lead
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # aux
        pred_scores_aux = pred_scores_aux.permute(0, 2, 1).contiguous()
        pred_distri_aux = pred_distri_aux.permute(0, 2, 1).contiguous()


        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        # make anchor
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        anchor_points_aux, stride_tensor_aux = make_anchors(feats_aux, self.stride, 0.5)
    

        # targets aux
        targets_aux = self.preprocess(targets[:, [0, 1, 6, 7, 8, 9]], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels_aux, gt_bboxes_aux = targets_aux.split((1, 4), 2)  # cls, xyxy
        mask_gt_aux = gt_bboxes_aux.sum(2, keepdim=True).gt_(0)

        # targets
        targets = self.preprocess(targets[:,:6], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)
        
        
        # pboxes_aux
        pred_bboxes_aux = self.bbox_decode(anchor_points_aux, pred_distri_aux)  # xyxy, (b, h*w, 4)
        target_labels_aux, target_bboxes_aux, target_scores_aux, fg_mask_aux = self.assigner_aux(
            pred_scores_aux.detach().sigmoid(),
            (pred_bboxes_aux.detach() * stride_tensor_aux).type(gt_bboxes_aux.dtype),
            anchor_points_aux * stride_tensor_aux,
            gt_labels_aux,
            gt_bboxes_aux,
            mask_gt_aux)
        
        target_bboxes_aux /= stride_tensor_aux
        target_scores_sum_aux = max(target_scores_aux.sum(), 1)

        # cls loss
        loss[1] = self.BCEcls(pred_scores_aux, target_scores_aux.to(dtype)).sum() / target_scores_sum_aux # BCE aux
        loss[1] *= 0.25
        loss[1] += self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # BCE lead

        # if target_scores_sum > 1:
        #     loss[1] /=  target_scores_sum # avoid devide zero error, devide by zero will cause loss to be inf or nan.

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2], iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)
        
        if fg_mask_aux.sum():
            loss_0, loss_2, iou_aux = self.bbox_loss_aux(pred_distri_aux,
                                                   pred_bboxes_aux,
                                                   anchor_points_aux,
                                                   target_bboxes_aux,
                                                   target_scores_aux,
                                                   target_scores_sum_aux,
                                                   fg_mask_aux)

            loss[0] += loss_0 * 0.25
            loss[2] += loss_2 * 0.25

        loss[0] *= 7.5  # box gain
        if self.hyp["fl_gamma"] > 0:
            loss[1] *= 4.0
        else:
            loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    

class ComputeLoss_aux_rep:
    # Compute losses
    def __init__(self, model, use_dfl=True, tal_topk=10, vis_guide=True, vis_thres=0.4):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            print("using FocalLoss")
            BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        # self.stride_aux = m.stride_aux if hasattr(m, 'stride_aux') else None

        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.device = device
        self.vis_guide = vis_guide
        
        if self.vis_guide:
            print('visible guide')
            self.assigner = TaskAlignedAssigner_Guided(topk=tal_topk,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)),
                                            vis_thres=vis_thres)
        else:
            self.assigner = TaskAlignedAssigner(topk=tal_topk,
                                                num_classes=self.nc,
                                                alpha=float(os.getenv('YOLOA', 0.5)),
                                                beta=float(os.getenv('YOLOB', 6.0)))
        
        self.assigner_aux = TaskAlignedAssigner(topk=tal_topk,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.bbox_loss_aux = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl
        
        
    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                
                if n:   
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy_v6(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, p, targets, img=None):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        lrepBox, lrepGT = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

        feats, pred_distri, pred_scores, feats_aux, pred_distri_aux, pred_scores_aux = p

        # lead
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # aux
        # rep_aux = True
        pred_scores_aux = pred_scores_aux.permute(0, 2, 1).contiguous()
        pred_distri_aux = pred_distri_aux.permute(0, 2, 1).contiguous()


        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        # make anchor
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        # make anchor aux
        # if self.stride_aux is not None:
        #     anchor_points_aux, stride_tensor_aux = make_anchors(feats_aux, self.stride_aux, 0.5)
        # else:
        #     anchor_points_aux, stride_tensor_aux = make_anchors(feats_aux, self.stride, 0.5)
        
        anchor_points_aux, stride_tensor_aux = make_anchors(feats_aux, self.stride, 0.5)

        # targets aux
        targets_aux = self.preprocess(targets[:, [0, 1, 6, 7, 8, 9]], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels_aux, gt_bboxes_aux = targets_aux.split((1, 4), 2)  # cls, xyxy
        mask_gt_aux = gt_bboxes_aux.sum(2, keepdim=True).gt_(0)

        # targets
        targets = self.preprocess(targets[:,:6], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)


        # pboxes_aux
        pred_bboxes_aux = self.bbox_decode(anchor_points_aux, pred_distri_aux)  # xyxy, (b, h*w, 4)
        target_labels_aux, target_bboxes_aux, target_scores_aux, fg_mask_aux = self.assigner_aux(
            pred_scores_aux.detach().sigmoid(),
            (pred_bboxes_aux.detach() * stride_tensor_aux).type(gt_bboxes_aux.dtype),
            anchor_points_aux * stride_tensor_aux,
            gt_labels_aux,
            gt_bboxes_aux,
            mask_gt_aux)
        
        target_bboxes_aux /= stride_tensor_aux
        target_scores_sum_aux = max(target_scores_aux.sum(), 1)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        
        if self.vis_guide:
            target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels_aux,
                gt_labels,
                gt_bboxes_aux,
                gt_bboxes,
                mask_gt_aux)

        else:
            target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # print(f'cls size: {pred_scores.size()}')
        # print(f'box size: {pred_distri.size()}')
        # # print(f'anchor size: {anchor_points.size()}')
        # # print(f'stride_tensor size: {stride_tensor.size()}')
        # print(f'pred_bboxes size: {pred_bboxes.size()}')
        # print(f'target box size: {target_bboxes.size()}')
        # exit()

        

        # if self.vis_guide:
        #     target_scores = torch.max(target_scores, target_scores_aux)
        #     target_scores_sum = max(target_scores.sum(), 1)
        
        # cls loss
        loss[1] = self.BCEcls(pred_scores_aux, target_scores_aux.to(dtype)).sum() / target_scores_sum_aux # BCE aux
        loss[1] *= 0.25
        loss[1] += self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # BCE lead

        # if target_scores_sum > 1:
        #     loss[1] /=  target_scores_sum # avoid devide zero error, devide by zero will cause loss to be inf or nan.

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2], iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)
            
            # Repulsion Loss
            bts = 0
            deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
            Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
            _lrepGT = 0.0
            _lrepBox = 0.0
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

            for indexs in range(0, batch_size):  # iterate batch 
                pred_bboxes_pos = torch.masked_select(pred_bboxes[indexs], bbox_mask[indexs]).view(-1, 4)
                target_bboxes_pos = torch.masked_select(target_bboxes[indexs], bbox_mask[indexs]).view(-1, 4)
                if pred_bboxes_pos.shape[0] != 0:
                    lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos, target_bboxes_pos, deta=deta, \
                                                           pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
                    _lrepGT += lrepgt
                    _lrepBox += lrepbox
                    bts += 1
            if bts > 0:
                _lrepGT /= bts
                _lrepBox /= bts
            lrepGT += _lrepGT
            lrepBox += _lrepBox
            
        
        if fg_mask_aux.sum():
            loss_0, loss_2, iou_aux = self.bbox_loss_aux(pred_distri_aux,
                                                   pred_bboxes_aux,
                                                   anchor_points_aux,
                                                   target_bboxes_aux,
                                                   target_scores_aux,
                                                   target_scores_sum_aux,
                                                   fg_mask_aux)

            loss[0] += loss_0 * 0.25
            loss[2] += loss_2 * 0.25

           
            # Repulsion Loss
            bts = 0
            deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
            Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
            _lrepGT = 0.0
            _lrepBox = 0.0
            bbox_mask_aux = fg_mask_aux.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

            for indexs in range(0, batch_size):  # iterate batch 
                pred_bboxes_pos_aux = torch.masked_select(pred_bboxes_aux[indexs], bbox_mask_aux[indexs]).view(-1, 4)
                target_bboxes_pos_aux = torch.masked_select(target_bboxes_aux[indexs], bbox_mask_aux[indexs]).view(-1, 4)
                if pred_bboxes_pos_aux.shape[0] != 0:
                    lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos_aux, target_bboxes_pos_aux, deta=deta, \
                                                        pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
                    _lrepGT += lrepgt
                    _lrepBox += lrepbox
                    bts += 1
            if bts > 0:
                _lrepGT /= bts
                _lrepBox /= bts
            lrepGT += _lrepGT * 0.25
            lrepBox += _lrepBox * 0.25

        # loss gain
        loss[0] *= 7.5  # box gain
        if self.hyp["fl_gamma"] > 0:
            loss[1] *= 2.5
        else:
            loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        lrep = self.hyp['alpha'] * lrepGT / 0.5 + self.hyp['beta'] * lrepBox / 0.5 # total repulsion loss


        return (loss.sum() + lrep) * batch_size , torch.cat((loss, lrep)).detach()  # loss(box, cls, dfl, repulsion)
    


    
class ComputeLoss_aux_rep_offset:
    # Compute losses
    def __init__(self, model, use_dfl=True, tal_topk=10):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            print("using FocalLoss")
            BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        # self.stride_aux = m.stride_aux if hasattr(m, 'stride_aux') else None

        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.device = device

        self.assigner = TaskAlignedAssigner(topk=tal_topk,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        
        self.assigner_aux = TaskAlignedAssigner(topk=tal_topk,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.bbox_loss_aux = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                
                if n:   
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy_v6(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist, offset):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return distoff2bbox(pred_dist, offset, anchor_points, xywh=False)

    def __call__(self, p, targets, img=None):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        lrepBox, lrepGT = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

        feats, pred_distri, pred_scores, feats_aux, pred_distri_aux, pred_scores_aux, offset, offset_aux = p

        # lead
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        offset = offset.permute(0, 2, 1).contiguous()

        # aux
        # rep_aux = True
        pred_scores_aux = pred_scores_aux.permute(0, 2, 1).contiguous()
        pred_distri_aux = pred_distri_aux.permute(0, 2, 1).contiguous()
        offset_aux = offset_aux.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        # make anchor
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        # make anchor aux
        # if self.stride_aux is not None:
        #     anchor_points_aux, stride_tensor_aux = make_anchors(feats_aux, self.stride_aux, 0.5)
        # else:
        #     anchor_points_aux, stride_tensor_aux = make_anchors(feats_aux, self.stride, 0.5)
        
        anchor_points_aux, stride_tensor_aux = make_anchors(feats_aux, self.stride, 0.5)

        # targets aux
        targets_aux = self.preprocess(targets[:, [0, 1, 6, 7, 8, 9]], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels_aux, gt_bboxes_aux = targets_aux.split((1, 4), 2)  # cls, xyxy
        mask_gt_aux = gt_bboxes_aux.sum(2, keepdim=True).gt_(0)

        # targets
        targets = self.preprocess(targets[:,:6], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, offset)  # xyxy, (b, h*w, 4)
        # print(f'cls size: {pred_scores.size()}')
        # print(f'box size: {pred_distri.size()}')
        # print(f'anchor size: {anchor_points.size()}')
        # print(f'stride_tensor size: {stride_tensor.size()}')
        # print(f'pred_bboxes size: {pred_bboxes.size()}')
        # exit()
        
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)
       
        # pboxes_aux
        pred_bboxes_aux = self.bbox_decode(anchor_points_aux, pred_distri_aux, offset_aux)  # xyxy, (b, h*w, 4)
        target_labels_aux, target_bboxes_aux, target_scores_aux, fg_mask_aux = self.assigner_aux(
            pred_scores_aux.detach().sigmoid(),
            (pred_bboxes_aux.detach() * stride_tensor_aux).type(gt_bboxes_aux.dtype),
            anchor_points_aux * stride_tensor_aux,
            gt_labels_aux,
            gt_bboxes_aux,
            mask_gt_aux)
        
        target_bboxes_aux /= stride_tensor_aux
        target_scores_sum_aux = max(target_scores_aux.sum(), 1)

        # cls loss
        loss[1] = self.BCEcls(pred_scores_aux, target_scores_aux.to(dtype)).sum() / target_scores_sum_aux # BCE aux
        loss[1] *= 0.25
        loss[1] += self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # BCE lead

        # if target_scores_sum > 1:
        #     loss[1] /=  target_scores_sum # avoid devide zero error, devide by zero will cause loss to be inf or nan.

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2], iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)
            
            # Repulsion Loss
            bts = 0
            deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
            Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
            _lrepGT = 0.0
            _lrepBox = 0.0
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

            for indexs in range(0, batch_size):  # iterate batch 
                pred_bboxes_pos = torch.masked_select(pred_bboxes[indexs], bbox_mask[indexs]).view(-1, 4)
                target_bboxes_pos = torch.masked_select(target_bboxes[indexs], bbox_mask[indexs]).view(-1, 4)
                if pred_bboxes_pos.shape[0] != 0:
                    lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos, target_bboxes_pos, deta=deta, \
                                                           pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
                    _lrepGT += lrepgt
                    _lrepBox += lrepbox
                    bts += 1
            if bts > 0:
                _lrepGT /= bts
                _lrepBox /= bts
            lrepGT += _lrepGT
            lrepBox += _lrepBox
            
        
        if fg_mask_aux.sum():
            loss_0, loss_2, iou_aux = self.bbox_loss_aux(pred_distri_aux,
                                                   pred_bboxes_aux,
                                                   anchor_points_aux,
                                                   target_bboxes_aux,
                                                   target_scores_aux,
                                                   target_scores_sum_aux,
                                                   fg_mask_aux)

            loss[0] += loss_0 * 0.25
            loss[2] += loss_2 * 0.25

           
            # Repulsion Loss
            bts = 0
            deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
            Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
            _lrepGT = 0.0
            _lrepBox = 0.0
            bbox_mask_aux = fg_mask_aux.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

            for indexs in range(0, batch_size):  # iterate batch 
                pred_bboxes_pos_aux = torch.masked_select(pred_bboxes_aux[indexs], bbox_mask_aux[indexs]).view(-1, 4)
                target_bboxes_pos_aux = torch.masked_select(target_bboxes_aux[indexs], bbox_mask_aux[indexs]).view(-1, 4)
                if pred_bboxes_pos_aux.shape[0] != 0:
                    lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos_aux, target_bboxes_pos_aux, deta=deta, \
                                                        pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
                    _lrepGT += lrepgt
                    _lrepBox += lrepbox
                    bts += 1
            if bts > 0:
                _lrepGT /= bts
                _lrepBox /= bts
            lrepGT += _lrepGT * 0.25
            lrepBox += _lrepBox * 0.25

        
        # loss gain
        loss[0] *= 7.5  # box gain
        if self.hyp["fl_gamma"] > 0:
            loss[1] *= 2.5
        else:
            loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        lrep = self.hyp['alpha'] * lrepGT / 0.5 + self.hyp['beta'] * lrepBox / 0.5 # total repulsion loss


        return (loss.sum() + lrep) * batch_size , torch.cat((loss, lrep)).detach()  # loss(box, cls, dfl, repulsion)
    

    

class ComputeLoss_aux_end2end:
    # Compute losses
    def __init__(self, model, use_dfl=True, tal_topk=10):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.device = device

        self.assigner = TaskAlignedAssigner(topk=tal_topk,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        
        self.assigner_aux = TaskAlignedAssigner(topk=tal_topk,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        
        self.assigner_e2e = TaskAlignedAssigner(topk=1,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.bbox_loss_aux = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.bbox_loss_e2e = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                
                if n:   
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy_v6(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, p, targets, img=None):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        feats_e2e, pred_distri_e2e, pred_scores_e2e, feats, pred_distri, pred_scores, feats_aux, pred_distri_aux, pred_scores_aux = p

        # lead -> end2end
        pred_scores_e2e = pred_scores_e2e.permute(0, 2, 1).contiguous()
        pred_distri_e2e = pred_distri_e2e.permute(0, 2, 1).contiguous()

        # aux full
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # aux vis
        rep_aux = True
        pred_scores_aux = pred_scores_aux.permute(0, 2, 1).contiguous()
        pred_distri_aux = pred_distri_aux.permute(0, 2, 1).contiguous()


        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats_e2e[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        # make anchor
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        anchor_points_aux, stride_tensor_aux = make_anchors(feats_aux, self.stride, 0.5)
        anchor_points_e2e, stride_tensor_e2e = make_anchors(feats_e2e, self.stride, 0.5)

        # targets aux vis
        targets_aux = self.preprocess(targets[:, [0, 1, 6, 7, 8, 9]], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels_aux, gt_bboxes_aux = targets_aux.split((1, 4), 2)  # cls, xyxy
        mask_gt_aux = gt_bboxes_aux.sum(2, keepdim=True).gt_(0)

        # targets lead full end2end
        targets_e2e = self.preprocess(targets[:,:6], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels_e2e, gt_bboxes_e2e = targets_e2e.split((1, 4), 2)  # cls, xyxy
        mask_gt_e2e= gt_bboxes_e2e.sum(2, keepdim=True).gt_(0)

        # targets aux full
        targets = self.preprocess(targets[:,:6], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        
        # pboxes lead end2end
        pred_bboxes_e2e = self.bbox_decode(anchor_points_e2e, pred_distri_e2e)  # xyxy, (b, h*w, 4)
        target_labels_e2e, target_bboxes_e2e, target_scores_e2e, fg_mask_e2e = self.assigner_e2e(
            pred_scores_e2e.detach().sigmoid(),
            (pred_bboxes_e2e.detach() * stride_tensor_e2e).type(gt_bboxes_e2e.dtype),
            anchor_points_e2e * stride_tensor_e2e,
            gt_labels_e2e,
            gt_bboxes_e2e,
            mask_gt_e2e)

        target_bboxes_e2e /= stride_tensor_e2e
        target_scores_sum_e2e = max(target_scores_e2e.sum(), 1)


        # pboxes aux full
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)
        
        # pboxes aux vis
        pred_bboxes_aux = self.bbox_decode(anchor_points_aux, pred_distri_aux)  # xyxy, (b, h*w, 4)
        target_labels_aux, target_bboxes_aux, target_scores_aux, fg_mask_aux = self.assigner_aux(
            pred_scores_aux.detach().sigmoid(),
            (pred_bboxes_aux.detach() * stride_tensor_aux).type(gt_bboxes_aux.dtype),
            anchor_points_aux * stride_tensor_aux,
            gt_labels_aux,
            gt_bboxes_aux,
            mask_gt_aux)
        
        target_bboxes_aux /= stride_tensor_aux
        target_scores_sum_aux = max(target_scores_aux.sum(), 1)

        # cls loss
        loss[1] = self.BCEcls(pred_scores_aux, target_scores_aux.to(dtype)).sum() / target_scores_sum_aux # BCE aux vis
        loss[1] *= 0.25
        loss[1] += 0.5 * self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # BCE aux full
        loss[1] += self.BCEcls(pred_scores_e2e, target_scores_e2e.to(dtype)).sum() / target_scores_sum_e2e # BCE lead e2e

        # if target_scores_sum > 1:
        #     loss[1] /=  target_scores_sum # avoid devide zero error, devide by zero will cause loss to be inf or nan.

        # bbox loss 

        # lead end2end
        if fg_mask_e2e.sum():
            loss[0], loss[2], iou = self.bbox_loss_e2e(pred_distri_e2e,
                                                   pred_bboxes_e2e,
                                                   anchor_points_e2e,
                                                   target_bboxes_e2e,
                                                   target_scores_e2e,
                                                   target_scores_sum_e2e,
                                                   fg_mask_e2e)
            
        # aux full
        if fg_mask.sum():
            loss_0, loss_2, iou_a = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)
            
            loss[0] += loss_0 * 0.5
            loss[2] += loss_2 * 0.5


        # aux vis
        if fg_mask_aux.sum():
            loss_0_aux, loss_2_aux, iou_aux = self.bbox_loss_aux(pred_distri_aux,
                                                   pred_bboxes_aux,
                                                   anchor_points_aux,
                                                   target_bboxes_aux,
                                                   target_scores_aux,
                                                   target_scores_sum_aux,
                                                   fg_mask_aux)

            loss[0] += loss_0_aux * 0.25
            loss[2] += loss_2_aux * 0.25


        # loss gain
        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain

        return loss.sum() * batch_size , loss.detach()  # loss(box, cls, dfl, repulsion)
    

# class ComputeLoss_aux_rep_end2end:
#     # Compute losses
#     def __init__(self, model, use_dfl=True, tal_topk=10):
#         device = next(model.parameters()).device  # get model device
#         h = model.hyp  # hyperparameters

#         # Define criteria
#         BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

#         # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

#         # Focal loss
#         g = h["fl_gamma"]  # focal loss gamma
#         if g > 0:
#             BCEcls = FocalLoss(BCEcls, g)

#         m = de_parallel(model).model[-1]  # Detect() module
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
#         self.BCEcls = BCEcls
#         self.hyp = h
#         self.stride = m.stride  # model strides
#         self.nc = m.nc  # number of classes
#         self.nl = m.nl  # number of layers
#         self.device = device

#         self.assigner = TaskAlignedAssigner(topk=tal_topk,
#                                             num_classes=self.nc,
#                                             alpha=float(os.getenv('YOLOA', 0.5)),
#                                             beta=float(os.getenv('YOLOB', 6.0)))
        
#         self.assigner_aux = TaskAlignedAssigner(topk=tal_topk,
#                                             num_classes=self.nc,
#                                             alpha=float(os.getenv('YOLOA', 0.5)),
#                                             beta=float(os.getenv('YOLOB', 6.0)))
        
#         self.assigner_e2e = TaskAlignedAssigner(topk=1,
#                                             num_classes=self.nc,
#                                             alpha=float(os.getenv('YOLOA', 0.5)),
#                                             beta=float(os.getenv('YOLOB', 6.0)))
        
#         self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
#         self.bbox_loss_aux = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
#         self.bbox_loss_e2e = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
#         self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
#         self.use_dfl = use_dfl

#     def preprocess(self, targets, batch_size, scale_tensor):
#         if targets.shape[0] == 0:
#             out = torch.zeros(batch_size, 0, 5, device=self.device)
#         else:
#             i = targets[:, 0]  # image index
#             _, counts = i.unique(return_counts=True)
#             out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
#             for j in range(batch_size):
#                 matches = i == j
#                 n = matches.sum()
                
#                 if n:   
#                     out[j, :n] = targets[matches, 1:]
#             out[..., 1:5] = xywh2xyxy_v6(out[..., 1:5].mul_(scale_tensor))
#         return out

#     def bbox_decode(self, anchor_points, pred_dist):
#         if self.use_dfl:
#             b, a, c = pred_dist.shape  # batch, anchors, channels
#             pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
#             # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
#             # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
#         return dist2bbox(pred_dist, anchor_points, xywh=False)

#     def __call__(self, p, targets, img=None):
#         loss = torch.zeros(3, device=self.device)  # box, cls, dfl
#         lrepBox, lrepGT = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

#         feats_e2e, pred_distri_e2e, pred_scores_e2e, feats, pred_distri, pred_scores, feats_aux, pred_distri_aux, pred_scores_aux = p

#         # lead -> end2end
#         pred_scores_e2e = pred_scores_e2e.permute(0, 2, 1).contiguous()
#         pred_distri_e2e = pred_distri_e2e.permute(0, 2, 1).contiguous()

#         # aux full
#         pred_scores = pred_scores.permute(0, 2, 1).contiguous()
#         pred_distri = pred_distri.permute(0, 2, 1).contiguous()

#         # aux vis
#         rep_aux = True
#         pred_scores_aux = pred_scores_aux.permute(0, 2, 1).contiguous()
#         pred_distri_aux = pred_distri_aux.permute(0, 2, 1).contiguous()


#         dtype = pred_scores.dtype
#         batch_size, grid_size = pred_scores.shape[:2]
#         imgsz = torch.tensor(feats_e2e[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

#         # make anchor
#         anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
#         anchor_points_aux, stride_tensor_aux = make_anchors(feats_aux, self.stride, 0.5)
#         anchor_points_e2e, stride_tensor_e2e = make_anchors(feats_e2e, self.stride, 0.5)

#         # targets aux vis
#         targets_aux = self.preprocess(targets[:, [0, 1, 6, 7, 8, 9]], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
#         gt_labels_aux, gt_bboxes_aux = targets_aux.split((1, 4), 2)  # cls, xyxy
#         mask_gt_aux = gt_bboxes_aux.sum(2, keepdim=True).gt_(0)

#         # targets lead full end2end
#         targets_e2e = self.preprocess(targets[:,:6], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
#         gt_labels_e2e, gt_bboxes_e2e = targets_e2e.split((1, 4), 2)  # cls, xyxy
#         mask_gt_e2e= gt_bboxes_e2e.sum(2, keepdim=True).gt_(0)

#         # targets aux full
#         targets = self.preprocess(targets[:,:6], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
#         gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
#         mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        
#         # pboxes lead end2end
#         pred_bboxes_e2e = self.bbox_decode(anchor_points_e2e, pred_distri_e2e)  # xyxy, (b, h*w, 4)
#         target_labels_e2e, target_bboxes_e2e, target_scores_e2e, fg_mask_e2e = self.assigner_e2e(
#             pred_scores_e2e.detach().sigmoid(),
#             (pred_bboxes_e2e.detach() * stride_tensor_e2e).type(gt_bboxes_e2e.dtype),
#             anchor_points_e2e * stride_tensor_e2e,
#             gt_labels_e2e,
#             gt_bboxes_e2e,
#             mask_gt_e2e)

#         target_bboxes_e2e /= stride_tensor_e2e
#         target_scores_sum_e2e = max(target_scores_e2e.sum(), 1)


#         # pboxes aux full
#         pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
#         target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
#             pred_scores.detach().sigmoid(),
#             (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
#             anchor_points * stride_tensor,
#             gt_labels,
#             gt_bboxes,
#             mask_gt)

#         target_bboxes /= stride_tensor
#         target_scores_sum = max(target_scores.sum(), 1)
        
#         # pboxes aux vis
#         pred_bboxes_aux = self.bbox_decode(anchor_points_aux, pred_distri_aux)  # xyxy, (b, h*w, 4)
#         target_labels_aux, target_bboxes_aux, target_scores_aux, fg_mask_aux = self.assigner_aux(
#             pred_scores_aux.detach().sigmoid(),
#             (pred_bboxes_aux.detach() * stride_tensor_aux).type(gt_bboxes_aux.dtype),
#             anchor_points_aux * stride_tensor_aux,
#             gt_labels_aux,
#             gt_bboxes_aux,
#             mask_gt_aux)
        
#         target_bboxes_aux /= stride_tensor_aux
#         target_scores_sum_aux = max(target_scores_aux.sum(), 1)

#         # cls loss
#         loss[1] = self.BCEcls(pred_scores_aux, target_scores_aux.to(dtype)).sum() / target_scores_sum_aux # BCE aux vis
#         loss[1] *= 0.25
#         loss[1] += 0.5 * self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # BCE aux full
#         loss[1] += self.BCEcls(pred_scores_e2e, target_scores_e2e.to(dtype)).sum() / target_scores_sum_e2e # BCE lead e2e

#         # if target_scores_sum > 1:
#         #     loss[1] /=  target_scores_sum # avoid devide zero error, devide by zero will cause loss to be inf or nan.

#         # bbox loss 

#         # lead end2end
#         if fg_mask_e2e.sum():
#             loss[0], loss[2], iou = self.bbox_loss_e2e(pred_distri_e2e,
#                                                    pred_bboxes_e2e,
#                                                    anchor_points_e2e,
#                                                    target_bboxes_e2e,
#                                                    target_scores_e2e,
#                                                    target_scores_sum_e2e,
#                                                    fg_mask_e2e)
            
#             # Repulsion Loss
#             bts = 0
#             deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
#             Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
#             _lrepGT = 0.0
#             _lrepBox = 0.0
#             bbox_mask_e2e = fg_mask_e2e.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

#             for indexs in range(0, batch_size):  # iterate batch 
#                 pred_bboxes_pos = torch.masked_select(pred_bboxes_e2e[indexs], bbox_mask_e2e[indexs]).view(-1, 4)
#                 target_bboxes_pos = torch.masked_select(target_bboxes_e2e[indexs], bbox_mask_e2e[indexs]).view(-1, 4)
#                 if pred_bboxes_pos.shape[0] != 0:
#                     lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos, target_bboxes_pos, deta=deta, \
#                                                            pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
#                     _lrepGT += lrepgt
#                     _lrepBox += lrepbox
#                     bts += 1
#             if bts > 0:
#                 _lrepGT /= bts
#                 _lrepBox /= bts
#             lrepGT += _lrepGT
#             lrepBox += _lrepBox

#         # aux full
#         if fg_mask.sum():
#             loss_0, loss_2, iou_a = self.bbox_loss(pred_distri,
#                                                    pred_bboxes,
#                                                    anchor_points,
#                                                    target_bboxes,
#                                                    target_scores,
#                                                    target_scores_sum,
#                                                    fg_mask)
            
#             loss[0] += loss_0 * 0.5
#             loss[2] += loss_2 * 0.5

#             # Repulsion Loss
#             bts = 0
#             deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
#             Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
#             _lrepGT_1 = 0.0
#             _lrepBox_1 = 0.0
#             bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

#             for indexs in range(0, batch_size):  # iterate batch 
#                 pred_bboxes_pos = torch.masked_select(pred_bboxes[indexs], bbox_mask[indexs]).view(-1, 4)
#                 target_bboxes_pos = torch.masked_select(target_bboxes[indexs], bbox_mask[indexs]).view(-1, 4)
#                 if pred_bboxes_pos.shape[0] != 0:
#                     lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos, target_bboxes_pos, deta=deta, \
#                                                            pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
#                     _lrepGT_1 += lrepgt
#                     _lrepBox_1 += lrepbox
#                     bts += 1
#             if bts > 0:
#                 _lrepGT_1 /= bts
#                 _lrepBox_1 /= bts
#             lrepGT += _lrepGT_1 * 0.5
#             lrepBox += _lrepBox_1 * 0.5
            
#         # aux vis
#         if fg_mask_aux.sum():
#             loss_0_aux, loss_2_aux, iou_aux = self.bbox_loss_aux(pred_distri_aux,
#                                                    pred_bboxes_aux,
#                                                    anchor_points_aux,
#                                                    target_bboxes_aux,
#                                                    target_scores_aux,
#                                                    target_scores_sum_aux,
#                                                    fg_mask_aux)

#             loss[0] += loss_0_aux * 0.25
#             loss[2] += loss_2_aux * 0.25

#             if rep_aux:
#                 # Repulsion Loss
#                 bts = 0
#                 deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
#                 Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
#                 _lrepGT = 0.0
#                 _lrepBox = 0.0
#                 bbox_mask_aux = fg_mask_aux.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

#                 for indexs in range(0, batch_size):  # iterate batch 
#                     pred_bboxes_pos_aux = torch.masked_select(pred_bboxes_aux[indexs], bbox_mask_aux[indexs]).view(-1, 4)
#                     target_bboxes_pos_aux = torch.masked_select(target_bboxes_aux[indexs], bbox_mask_aux[indexs]).view(-1, 4)
#                     if pred_bboxes_pos_aux.shape[0] != 0:
#                         lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos_aux, target_bboxes_pos_aux, deta=deta, \
#                                                             pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
#                         _lrepGT += lrepgt
#                         _lrepBox += lrepbox
#                         bts += 1
#                 if bts > 0:
#                     _lrepGT /= bts
#                     _lrepBox /= bts
#                 lrepGT += _lrepGT * 0.25
#                 lrepBox += _lrepBox * 0.25

        
#         # loss gain
#         loss[0] *= 7.5  # box gain
#         loss[1] *= 0.5  # cls gain
#         loss[2] *= 1.5  # dfl gain
#         lrep = self.hyp['alpha'] * lrepGT / 0.5 + self.hyp['beta'] * lrepBox / 0.5 # total repulsion loss


#         return (loss.sum() + lrep) * batch_size , torch.cat((loss, lrep)).detach()  # loss(box, cls, dfl, repulsion)
    


class ComputeLoss_aux_rep_triple:
    # Compute losses
    def __init__(self, model, use_dfl=True, tal_topk=10, vis_guide=True, vis_thres=0.4):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.device = device
        self.vis_guide = vis_guide

        if self.vis_guide:
            print('use visual guided label assigner')
            self.assigner = TaskAlignedAssigner_Guided(topk=10,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)),
                                            vis_thres=vis_thres)
            self.assigner_e2e = TaskAlignedAssigner_Guided(topk=tal_topk,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)),
                                            vis_thres=vis_thres)
        else:
            self.assigner = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        
            self.assigner_e2e = TaskAlignedAssigner(topk=tal_topk,
                                                num_classes=self.nc,
                                                alpha=float(os.getenv('YOLOA', 0.5)),
                                                beta=float(os.getenv('YOLOB', 6.0)))
            
        self.assigner_aux = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        
        
        
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.bbox_loss_aux = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.bbox_loss_e2e = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                
                if n:   
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy_v6(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, p, targets, img=None):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        lrepBox, lrepGT = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

        feats_e2e, pred_distri_e2e, pred_scores_e2e, feats, pred_distri, pred_scores, feats_aux, pred_distri_aux, pred_scores_aux = p

        # lead -> end2end
        pred_scores_e2e = pred_scores_e2e.permute(0, 2, 1).contiguous()
        pred_distri_e2e = pred_distri_e2e.permute(0, 2, 1).contiguous()

        # aux full
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # aux vis
        rep_aux = True
        pred_scores_aux = pred_scores_aux.permute(0, 2, 1).contiguous()
        pred_distri_aux = pred_distri_aux.permute(0, 2, 1).contiguous()


        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats_e2e[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        # make anchor
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        anchor_points_aux, stride_tensor_aux = make_anchors(feats_aux, self.stride, 0.5)
        anchor_points_e2e, stride_tensor_e2e = make_anchors(feats_e2e, self.stride, 0.5)

        # targets aux vis
        targets_aux = self.preprocess(targets[:, [0, 1, 6, 7, 8, 9]], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels_aux, gt_bboxes_aux = targets_aux.split((1, 4), 2)  # cls, xyxy
        mask_gt_aux = gt_bboxes_aux.sum(2, keepdim=True).gt_(0)

        # targets lead full end2end
        targets_e2e = self.preprocess(targets[:,:6], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels_e2e, gt_bboxes_e2e = targets_e2e.split((1, 4), 2)  # cls, xyxy
        mask_gt_e2e= gt_bboxes_e2e.sum(2, keepdim=True).gt_(0)

        # targets aux full
        targets = self.preprocess(targets[:,:6], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # lead 
        pred_bboxes_e2e = self.bbox_decode(anchor_points_e2e, pred_distri_e2e)  # xyxy, (b, h*w, 4)
        # vis aux
        pred_bboxes_aux = self.bbox_decode(anchor_points_aux, pred_distri_aux)  # xyxy, (b, h*w, 4)
        # full aux
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # pboxes lead end2end
        if self.vis_guide:
            target_labels_e2e, target_bboxes_e2e, target_scores_e2e, fg_mask_e2e = self.assigner_e2e(
                pred_scores_e2e.detach().sigmoid(),
                (pred_bboxes_e2e.detach() * stride_tensor_e2e).type(gt_bboxes_e2e.dtype),
                anchor_points_e2e * stride_tensor_e2e,
                gt_labels_aux,
                gt_labels_e2e,
                gt_bboxes_aux,
                gt_bboxes_e2e,
                mask_gt_aux)
        else:
            target_labels_e2e, target_bboxes_e2e, target_scores_e2e, fg_mask_e2e = self.assigner_e2e(
                pred_scores_e2e.detach().sigmoid(),
                (pred_bboxes_e2e.detach() * stride_tensor_e2e).type(gt_bboxes_e2e.dtype),
                anchor_points_e2e * stride_tensor_e2e,
                gt_labels_e2e,
                gt_bboxes_e2e,
                mask_gt_e2e)

        target_bboxes_e2e /= stride_tensor_e2e
        target_scores_sum_e2e = max(target_scores_e2e.sum(), 1)


        # pboxes aux full
        if self.vis_guide:
            target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels_aux,
                gt_labels,
                gt_bboxes_aux,
                gt_bboxes,
                mask_gt_aux)
        else:
            target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)
        
        # pboxes aux vis
        
        target_labels_aux, target_bboxes_aux, target_scores_aux, fg_mask_aux = self.assigner_aux(
            pred_scores_aux.detach().sigmoid(),
            (pred_bboxes_aux.detach() * stride_tensor_aux).type(gt_bboxes_aux.dtype),
            anchor_points_aux * stride_tensor_aux,
            gt_labels_aux,
            gt_bboxes_aux,
            mask_gt_aux)
        
        target_bboxes_aux /= stride_tensor_aux
        target_scores_sum_aux = max(target_scores_aux.sum(), 1)

        # cls loss
        loss[1] = self.BCEcls(pred_scores_aux, target_scores_aux.to(dtype)).sum() / target_scores_sum_aux # BCE aux vis
        loss[1] *= 0.125
        loss[1] += 0.25 * self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # BCE aux full
        loss[1] += self.BCEcls(pred_scores_e2e, target_scores_e2e.to(dtype)).sum() / target_scores_sum_e2e # BCE lead e2e

        # if target_scores_sum > 1:
        #     loss[1] /=  target_scores_sum # avoid devide zero error, devide by zero will cause loss to be inf or nan.

        # bbox loss 

        # lead end2end
        if fg_mask_e2e.sum():
            loss[0], loss[2], iou = self.bbox_loss_e2e(pred_distri_e2e,
                                                   pred_bboxes_e2e,
                                                   anchor_points_e2e,
                                                   target_bboxes_e2e,
                                                   target_scores_e2e,
                                                   target_scores_sum_e2e,
                                                   fg_mask_e2e)
            
            # Repulsion Loss
            bts = 0
            deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
            Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
            _lrepGT = 0.0
            _lrepBox = 0.0
            bbox_mask_e2e = fg_mask_e2e.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

            for indexs in range(0, batch_size):  # iterate batch 
                pred_bboxes_pos = torch.masked_select(pred_bboxes_e2e[indexs], bbox_mask_e2e[indexs]).view(-1, 4)
                target_bboxes_pos = torch.masked_select(target_bboxes_e2e[indexs], bbox_mask_e2e[indexs]).view(-1, 4)
                if pred_bboxes_pos.shape[0] != 0:
                    lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos, target_bboxes_pos, deta=deta, \
                                                           pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
                    _lrepGT += lrepgt
                    _lrepBox += lrepbox
                    bts += 1
            if bts > 0:
                _lrepGT /= bts
                _lrepBox /= bts
            lrepGT += _lrepGT
            lrepBox += _lrepBox

        # aux full
        if fg_mask.sum():
            loss_0, loss_2, iou_a = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)
            
            loss[0] += loss_0 * 0.25
            loss[2] += loss_2 * 0.25

            # Repulsion Loss
            bts = 0
            deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
            Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
            _lrepGT_1 = 0.0
            _lrepBox_1 = 0.0
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

            for indexs in range(0, batch_size):  # iterate batch 
                pred_bboxes_pos = torch.masked_select(pred_bboxes[indexs], bbox_mask[indexs]).view(-1, 4)
                target_bboxes_pos = torch.masked_select(target_bboxes[indexs], bbox_mask[indexs]).view(-1, 4)
                if pred_bboxes_pos.shape[0] != 0:
                    lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos, target_bboxes_pos, deta=deta, \
                                                           pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
                    _lrepGT_1 += lrepgt
                    _lrepBox_1 += lrepbox
                    bts += 1
            if bts > 0:
                _lrepGT_1 /= bts
                _lrepBox_1 /= bts
            lrepGT += _lrepGT_1 * 0.25
            lrepBox += _lrepBox_1 * 0.25
            
        # aux vis
        if fg_mask_aux.sum():
            loss_0_aux, loss_2_aux, iou_aux = self.bbox_loss_aux(pred_distri_aux,
                                                   pred_bboxes_aux,
                                                   anchor_points_aux,
                                                   target_bboxes_aux,
                                                   target_scores_aux,
                                                   target_scores_sum_aux,
                                                   fg_mask_aux)

            loss[0] += loss_0_aux * 0.125
            loss[2] += loss_2_aux * 0.125

            if rep_aux:
                # Repulsion Loss
                bts = 0
                deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
                Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
                _lrepGT = 0.0
                _lrepBox = 0.0
                bbox_mask_aux = fg_mask_aux.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

                for indexs in range(0, batch_size):  # iterate batch 
                    pred_bboxes_pos_aux = torch.masked_select(pred_bboxes_aux[indexs], bbox_mask_aux[indexs]).view(-1, 4)
                    target_bboxes_pos_aux = torch.masked_select(target_bboxes_aux[indexs], bbox_mask_aux[indexs]).view(-1, 4)
                    if pred_bboxes_pos_aux.shape[0] != 0:
                        lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos_aux, target_bboxes_pos_aux, deta=deta, \
                                                            pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
                        _lrepGT += lrepgt
                        _lrepBox += lrepbox
                        bts += 1
                if bts > 0:
                    _lrepGT /= bts
                    _lrepBox /= bts
                lrepGT += _lrepGT * 0.125
                lrepBox += _lrepBox * 0.125

        
        # loss gain
        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        lrep = self.hyp['alpha'] * lrepGT / 0.5 + self.hyp['beta'] * lrepBox / 0.5 # total repulsion loss


        return (loss.sum() + lrep) * batch_size , torch.cat((loss, lrep)).detach()  # loss(box, cls, dfl, repulsion)
    


class ComputeLoss_aux_rep_end2end_dual:
    # Compute losses
    def __init__(self, model, use_dfl=True, tal_topk=10, vis_guide=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            print("using FocalLoss")
            BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        # self.stride_aux = m.stride_aux if hasattr(m, 'stride_aux') else None

        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.device = device

        self.assigner = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        
        self.assigner_e2e = TaskAlignedAssigner(topk=1,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        
        self.assigner_aux = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.bbox_loss_e2e = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.bbox_loss_aux = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl

        print('end-to-end dual training')
    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                
                if n:   
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy_v6(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, p, targets, img=None):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        lrepBox, lrepGT = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

        feats, pred_distri, pred_scores, feats_aux, pred_distri_aux, pred_scores_aux = p

        # lead
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # aux
        # rep_aux = True
        pred_scores_aux = pred_scores_aux.permute(0, 2, 1).contiguous()
        pred_distri_aux = pred_distri_aux.permute(0, 2, 1).contiguous()


        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        # make anchor
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        
        anchor_points_aux, stride_tensor_aux = make_anchors(feats_aux, self.stride, 0.5)

        # targets aux
        targets_aux = self.preprocess(targets[:, [0, 1, 6, 7, 8, 9]], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels_aux, gt_bboxes_aux = targets_aux.split((1, 4), 2)  # cls, xyxy
        mask_gt_aux = gt_bboxes_aux.sum(2, keepdim=True).gt_(0)

        # targets
        targets = self.preprocess(targets[:,:6], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)



        # pboxes
        # both topk boxes and e2e box
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # e2e
        # pred_bboxes_e2e = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        target_labels_e2e, target_bboxes_e2e, target_scores_e2e, fg_mask_e2e = self.assigner_e2e(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt)

        target_bboxes_e2e /= stride_tensor
        target_scores_sum_e2e = max(target_scores_e2e.sum(), 1)

        # pboxes_aux
        pred_bboxes_aux = self.bbox_decode(anchor_points_aux, pred_distri_aux)  # xyxy, (b, h*w, 4)
        target_labels_aux, target_bboxes_aux, target_scores_aux, fg_mask_aux = self.assigner_aux(
            pred_scores_aux.detach().sigmoid(),
            (pred_bboxes_aux.detach() * stride_tensor_aux).type(gt_bboxes_aux.dtype),
            anchor_points_aux * stride_tensor_aux,
            gt_labels_aux,
            gt_bboxes_aux,
            mask_gt_aux)
        
        target_bboxes_aux /= stride_tensor_aux
        target_scores_sum_aux = max(target_scores_aux.sum(), 1)

        
        # cls loss
        loss[1] = self.BCEcls(pred_scores_aux, target_scores_aux.to(dtype)).sum() / target_scores_sum_aux # BCE aux
        loss[1] *= 0.25
        loss[1] += 0.5 * self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # BCE topk
        loss[1] += self.BCEcls(pred_scores, target_scores_e2e.to(dtype)).sum() / target_scores_sum_e2e # BCE topk
        # if target_scores_sum > 1:
        #     loss[1] /=  target_scores_sum # avoid devide zero error, devide by zero will cause loss to be inf or nan.

        # bbox loss
        if fg_mask_e2e.sum():
            loss[0], loss[2], iou_e2e = self.bbox_loss_e2e(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes_e2e,
                                                   target_scores_e2e,
                                                   target_scores_sum_e2e,
                                                   fg_mask_e2e)
            
            # Repulsion Loss
            bts = 0
            deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
            Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
            _lrepGT = 0.0
            _lrepBox = 0.0
            bbox_mask_e2e = fg_mask_e2e.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

            for indexs in range(0, batch_size):  # iterate batch 
                pred_bboxes_pos_e2e = torch.masked_select(pred_bboxes[indexs], bbox_mask_e2e[indexs]).view(-1, 4)
                target_bboxes_pos_e2e = torch.masked_select(target_bboxes_e2e[indexs], bbox_mask_e2e[indexs]).view(-1, 4)
                if pred_bboxes_pos_e2e.shape[0] != 0:
                    lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos_e2e, target_bboxes_pos_e2e, deta=deta, \
                                                           pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
                    _lrepGT += lrepgt
                    _lrepBox += lrepbox
                    bts += 1
            if bts > 0:
                _lrepGT /= bts
                _lrepBox /= bts
            lrepGT += _lrepGT
            lrepBox += _lrepBox
            
        
        if fg_mask.sum():
            loss_0, loss_2, iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)

            loss[0] += loss_0 * 0.5
            loss[2] += loss_2 * 0.5

           
            # Repulsion Loss
            bts = 0
            deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
            Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
            _lrepGT = 0.0
            _lrepBox = 0.0
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

            for indexs in range(0, batch_size):  # iterate batch 
                pred_bboxes_pos = torch.masked_select(pred_bboxes[indexs], bbox_mask[indexs]).view(-1, 4)
                target_bboxes_pos = torch.masked_select(target_bboxes[indexs], bbox_mask[indexs]).view(-1, 4)
                if pred_bboxes_pos.shape[0] != 0:
                    lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos, target_bboxes_pos, deta=deta, \
                                                        pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
                    _lrepGT += lrepgt
                    _lrepBox += lrepbox
                    bts += 1
            if bts > 0:
                _lrepGT /= bts
                _lrepBox /= bts
            lrepGT += _lrepGT * 0.5
            lrepBox += _lrepBox * 0.5
        
        if fg_mask_aux.sum():
            loss_0, loss_2, iou_aux = self.bbox_loss_aux(pred_distri_aux,
                                                   pred_bboxes_aux,
                                                   anchor_points_aux,
                                                   target_bboxes_aux,
                                                   target_scores_aux,
                                                   target_scores_sum_aux,
                                                   fg_mask_aux)

            loss[0] += loss_0 * 0.25
            loss[2] += loss_2 * 0.25

           
            # Repulsion Loss
            bts = 0
            deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
            Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
            _lrepGT = 0.0
            _lrepBox = 0.0
            bbox_mask_aux = fg_mask_aux.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

            for indexs in range(0, batch_size):  # iterate batch 
                pred_bboxes_pos_aux = torch.masked_select(pred_bboxes_aux[indexs], bbox_mask_aux[indexs]).view(-1, 4)
                target_bboxes_pos_aux = torch.masked_select(target_bboxes_aux[indexs], bbox_mask_aux[indexs]).view(-1, 4)
                if pred_bboxes_pos_aux.shape[0] != 0:
                    lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos_aux, target_bboxes_pos_aux, deta=deta, \
                                                        pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
                    _lrepGT += lrepgt
                    _lrepBox += lrepbox
                    bts += 1
            if bts > 0:
                _lrepGT /= bts
                _lrepBox /= bts
            lrepGT += _lrepGT * 0.25
            lrepBox += _lrepBox * 0.25

        # loss gain
        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        lrep = self.hyp['alpha'] * lrepGT / 0.5 + self.hyp['beta'] * lrepBox / 0.5 # total repulsion loss


        return (loss.sum() + lrep) * batch_size , torch.cat((loss, lrep)).detach()  # loss(box, cls, dfl, repulsion)
        # return loss.sum() * batch_size , loss.detach()  # loss(box, cls, dfl)
    

class ComputeLoss_KD:
    # Compute losses
    def __init__(self, model, use_dfl=True, tal_topk=10, vis_guide=True):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            print("using FocalLoss")
            BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        # self.stride_aux = m.stride_aux if hasattr(m, 'stride_aux') else None

        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.device = device
        self.vis_guide = vis_guide
        
        if self.vis_guide:
            print('visible guide')
            self.assigner = TaskAlignedAssigner_Guided(topk=tal_topk,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        else:
            self.assigner = TaskAlignedAssigner(topk=tal_topk,
                                                num_classes=self.nc,
                                                alpha=float(os.getenv('YOLOA', 0.5)),
                                                beta=float(os.getenv('YOLOB', 6.0)))
        
        self.assigner_aux = TaskAlignedAssigner(topk=tal_topk,
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        
        self.bbox_loss = BboxLoss_KD(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.bbox_loss_aux = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl
        
        
    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                
                if n:   
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy_v6(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, p, pt, targets, img=None):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        lrepBox, lrepGT = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        loss_KL = torch.zeros(1, device=self.device)

        feats, pred_distri, pred_scores, feats_aux, pred_distri_aux, pred_scores_aux = p

        _, pred_distri_tea, pred_scores_tea = pt
        # pred_distri_tea = pred_distri_tea.detach()
        # pred_scores_tea = pred_scores_tea.detach()
        # lead
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # aux
        # rep_aux = True
        pred_scores_aux = pred_scores_aux.permute(0, 2, 1).contiguous()
        pred_distri_aux = pred_distri_aux.permute(0, 2, 1).contiguous()

        # teacher
        pred_scores_tea = pred_scores_tea.permute(0, 2, 1).contiguous()
        pred_distri_tea = pred_distri_tea.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        # make anchor
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        # make anchor aux
        # if self.stride_aux is not None:
        #     anchor_points_aux, stride_tensor_aux = make_anchors(feats_aux, self.stride_aux, 0.5)
        # else:
        #     anchor_points_aux, stride_tensor_aux = make_anchors(feats_aux, self.stride, 0.5)
        
        anchor_points_aux, stride_tensor_aux = make_anchors(feats_aux, self.stride, 0.5)

        # targets aux
        targets_aux = self.preprocess(targets[:, [0, 1, 6, 7, 8, 9]], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels_aux, gt_bboxes_aux = targets_aux.split((1, 4), 2)  # cls, xyxy
        mask_gt_aux = gt_bboxes_aux.sum(2, keepdim=True).gt_(0)

        # targets
        targets = self.preprocess(targets[:,:6], batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)


        # pboxes_aux
        pred_bboxes_aux = self.bbox_decode(anchor_points_aux, pred_distri_aux)  # xyxy, (b, h*w, 4)
        target_labels_aux, target_bboxes_aux, target_scores_aux, fg_mask_aux = self.assigner_aux(
            pred_scores_aux.detach().sigmoid(),
            (pred_bboxes_aux.detach() * stride_tensor_aux).type(gt_bboxes_aux.dtype),
            anchor_points_aux * stride_tensor_aux,
            gt_labels_aux,
            gt_bboxes_aux,
            mask_gt_aux)
        
        target_bboxes_aux /= stride_tensor_aux
        target_scores_sum_aux = max(target_scores_aux.sum(), 1)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        
        if self.vis_guide:
            target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels_aux,
                gt_labels,
                gt_bboxes_aux,
                gt_bboxes,
                mask_gt_aux)

        else:
            target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # teacher box
        pred_bboxes_tea = self.bbox_decode(anchor_points, pred_distri_tea)  # xyxy, (b, h*w, 4)


        # cls loss
        loss[1] = self.BCEcls(pred_scores_aux, target_scores_aux.to(dtype)).sum() / target_scores_sum_aux # BCE aux
        loss[1] *= 0.25
        loss[1] += self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # BCE lead
        # loss[1] += self.BCEcls(pred_scores, pred_scores_tea).sum()

        # if target_scores_sum > 1:
        #     loss[1] /=  target_scores_sum # avoid devide zero error, devide by zero will cause loss to be inf or nan.

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2], loss_iou_KD, loss_KL, iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask,
                                                   pred_distri_tea.detach(),
                                                   pred_bboxes_tea.detach())
            
            loss[0] += loss_iou_KD
            # Repulsion Loss
            bts = 0
            deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
            Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
            _lrepGT = 0.0
            _lrepBox = 0.0
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

            for indexs in range(0, batch_size):  # iterate batch 
                pred_bboxes_pos = torch.masked_select(pred_bboxes[indexs], bbox_mask[indexs]).view(-1, 4)
                target_bboxes_pos = torch.masked_select(target_bboxes[indexs], bbox_mask[indexs]).view(-1, 4)
                if pred_bboxes_pos.shape[0] != 0:
                    lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos, target_bboxes_pos, deta=deta, \
                                                           pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
                    _lrepGT += lrepgt
                    _lrepBox += lrepbox
                    bts += 1
            if bts > 0:
                _lrepGT /= bts
                _lrepBox /= bts
            lrepGT += _lrepGT
            lrepBox += _lrepBox
            
        
        if fg_mask_aux.sum():
            loss_0, loss_2, iou_aux = self.bbox_loss_aux(pred_distri_aux,
                                                   pred_bboxes_aux,
                                                   anchor_points_aux,
                                                   target_bboxes_aux,
                                                   target_scores_aux,
                                                   target_scores_sum_aux,
                                                   fg_mask_aux)

            loss[0] += loss_0 * 0.25
            loss[2] += loss_2 * 0.25

           
            # Repulsion Loss
            bts = 0
            deta = self.hyp['deta'] if self.hyp['deta'] else 0.5
            Rp_nms = self.hyp['Rp_nms'] if self.hyp['Rp_nms'] else 0.1
            _lrepGT = 0.0
            _lrepBox = 0.0
            bbox_mask_aux = fg_mask_aux.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)

            for indexs in range(0, batch_size):  # iterate batch 
                pred_bboxes_pos_aux = torch.masked_select(pred_bboxes_aux[indexs], bbox_mask_aux[indexs]).view(-1, 4)
                target_bboxes_pos_aux = torch.masked_select(target_bboxes_aux[indexs], bbox_mask_aux[indexs]).view(-1, 4)
                if pred_bboxes_pos_aux.shape[0] != 0:
                    lrepgt, lrepbox = repulsion_loss_torch(pred_bboxes_pos_aux, target_bboxes_pos_aux, deta=deta, \
                                                        pnms=Rp_nms, gtnms=Rp_nms, x1x2y1y2=True)
                    _lrepGT += lrepgt
                    _lrepBox += lrepbox
                    bts += 1
            if bts > 0:
                _lrepGT /= bts
                _lrepBox /= bts
            lrepGT += _lrepGT * 0.25
            lrepBox += _lrepBox * 0.25

        # loss gain
        loss[0] *= 7.5  # box gain
        if self.hyp["fl_gamma"] > 0:
            loss[1] *= 2.5
        else:
            loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        lrep = self.hyp['alpha'] * lrepGT / 0.5 + self.hyp['beta'] * lrepBox / 0.5 # total repulsion loss


        # print(loss.size(), lrep.size(), loss_KL.size())
        return (loss.sum() + lrep + loss_KL) * batch_size , torch.cat((loss, lrep, loss_KL)).detach()  # loss(box, cls, dfl, repulsion)