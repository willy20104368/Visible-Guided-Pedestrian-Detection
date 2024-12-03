import torch
import numpy as np
from utils.general import box_iou, box_iou_v5
from utils.general import bbox_iou, bbox_alpha_iou, box_iou, box_giou, box_diou, box_ciou, xywh2xyxy
import torch.nn.functional as F


# reference: https://github.com/dongdonghy/repulsion_loss_pytorch/blob/master/repulsion_loss.py
def IoG(gt_box, pre_box):
    inter_xmin = torch.max(gt_box[:, 0], pre_box[:, 0])
    inter_ymin = torch.max(gt_box[:, 1], pre_box[:, 1])
    inter_xmax = torch.min(gt_box[:, 2], pre_box[:, 2])
    inter_ymax = torch.min(gt_box[:, 3], pre_box[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = ((gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])).clamp(1e-6)
    return I / G

def smooth_ln(x, deta=0.5):
    return torch.where(
        torch.le(x, deta),
        -torch.log(1 - x),
        ((x - deta) / (1 - deta)) - np.log(1 - deta)
    )


def repulsion_loss_torch(pbox, gtbox, deta=0.5, pnms=0.1, gtnms=0.1, x1x2y1y2=False):
    repgt_loss = 0.0
    repbox_loss = 0.0
    pbox = pbox.detach()
    gtbox = gtbox.detach()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gtbox_cpu = gtbox.cuda().data.cpu().numpy()
    pgiou = box_iou_v5(pbox, gtbox, x1y1x2y2=x1x2y1y2)
    pgiou = pgiou.cuda().data.cpu().numpy()
    ppiou = box_iou_v5(pbox, pbox, x1y1x2y2=x1x2y1y2)
    ppiou = ppiou.cuda().data.cpu().numpy()
    # t1 = time.time()
    len = pgiou.shape[0]
    for j in range(len):
        for z in range(j, len):
            ppiou[j, z] = 0
            # if int(torch.sum(gtbox[j] == gtbox[z])) == 4:
            # if int(torch.sum(gtbox_cpu[j] == gtbox_cpu[z])) == 4:
            # if int(np.sum(gtbox_numpy[j] == gtbox_numpy[z])) == 4:
            if (gtbox_cpu[j][0]==gtbox_cpu[z][0]) and (gtbox_cpu[j][1]==gtbox_cpu[z][1]) and (gtbox_cpu[j][2]==gtbox_cpu[z][2]) and (gtbox_cpu[j][3]==gtbox_cpu[z][3]):
                pgiou[j, z] = 0
                pgiou[z, j] = 0
                ppiou[z, j] = 0

    # t2 = time.time()
    # print("for cycle cost time is: ", t2 - t1, "s")
    pgiou = torch.from_numpy(pgiou).cuda().detach()
    ppiou = torch.from_numpy(ppiou).cuda().detach()
    # repgt
    max_iou, argmax_iou = torch.max(pgiou, 1)
    pg_mask = torch.gt(max_iou, gtnms)
    num_repgt = pg_mask.sum()
    if num_repgt > 0:
        iou_pos = pgiou[pg_mask, :]
        max_iou_sec, argmax_iou_sec = torch.max(iou_pos, 1)
        pbox_sec = pbox[pg_mask, :]
        gtbox_sec = gtbox[argmax_iou_sec, :]
        IOG = IoG(gtbox_sec, pbox_sec)
        repgt_loss = smooth_ln(IOG, deta)
        repgt_loss = repgt_loss.mean()

    # repbox
    pp_mask = torch.gt(ppiou, pnms)  # 防止nms为0, 因为如果为0,那么上面的for循环就没有意义了 [N x N] error
    num_pbox = pp_mask.sum()
    if num_pbox > 0:
        repbox_loss = smooth_ln(ppiou, deta)
        repbox_loss = repbox_loss.mean()
    # mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
    # print(mem)
    torch.cuda.empty_cache()

    return repgt_loss, repbox_loss

def contrastive_loss(positives, negatives, temperature=0.1):

    pos_sim = F.cosine_similarity(positives.unsqueeze(1), positives.unsqueeze(0), dim=-1)
    neg_sim = F.cosine_similarity(positives.unsqueeze(1), negatives.unsqueeze(0), dim=-1)

    loss = -torch.log(torch.exp(pos_sim / temperature) / (torch.exp(pos_sim / temperature) + torch.sum(torch.exp(neg_sim / temperature), dim=-1))).mean()
    return loss

# def Cos_triplet_loss(a, p, n, temperature=0.07):
#     a_p = a.expand(p.size(0), -1)
#     a_n = a.expand(n.size(0), -1)

#     pos_sim = torch.nn.CosineEmbeddingLoss(margin=0.2, reduction='sum')(a_p, p, torch.ones(p.size(0)).cuda())
#     neg_sim = torch.nn.CosineEmbeddingLoss(margin=0.2, reduction='sum')(a_n, n, -torch.ones(n.size(0)).cuda())
#     # loss = -torch.log(torch.exp(pos_sim / temperature) / (torch.exp(pos_sim / temperature) + torch.exp(neg_sim / temperature)))

#     return pos_sim + neg_sim

def Cos_triplet_loss(a, p, n, temperature=0.07):

    pos_sim = F.cosine_similarity(a.unsqueeze(1), p.unsqueeze(0), dim=-1)
    neg_sim = F.cosine_similarity(a.unsqueeze(1), n.unsqueeze(0), dim=-1)
    loss = -torch.log(torch.exp(pos_sim / temperature) / (torch.exp(pos_sim / temperature) + torch.sum(torch.exp(neg_sim / temperature), dim=-1))).mean()

    return loss

# v1 small size of negative sample
def attribute_loss(gtbox, attr):
    gtbox = gtbox.detach()
    attr_loss = 0.0
    L = len(gtbox)
    count = 0
    for i in range(L):
            mask = gtbox == gtbox[i]
            mask[i] = False
            positive = attr[mask.any(dim=1)]
            mask[i] = True
            negative = attr[~mask.any(dim=1)]
            # attr_loss += Cos_triplet_loss(attr[i].unsqueeze(0), positive, negative)
            # if there has positive sample and negative sample 
            if positive.size(0) > 0 and negative.size(0) > 0:
                count += 1
                attr_loss += Cos_triplet_loss(attr[i].unsqueeze(0), positive, negative)
    return attr_loss / count if count > 0 else 1e-8
    # return attr_loss / L + 1e-8

# v2
# def attribute_loss_v2(gtbox, attr, bg, occ_ratio, temperature=0.14):
#     gtbox = gtbox.detach()
#     attr_loss = 0.0
#     L = len(gtbox)
#     count = 0
#     bg = bg[torch.isin(bg, attr,invert=True).all(dim=1)]
#     # print('attr:',attr.size())
#     # print('bg:',bg.size())
    
#     for i in range(L):
#             mask = gtbox == gtbox[i]
#             mask[i] = False
#             positive = attr[mask.any(dim=1)]
#             mask[i] = True
#             negative = torch.cat((attr[~mask.any(dim=1)], bg), 0)
#             # print(positive.size())
#             # print(negative.size())
#             # print("======================")
#             if positive.size(0) > 0:
#                 count += 1
#                 attr_loss += Cos_triplet_loss(attr[i].unsqueeze(0), positive, negative, temperature=temperature)

#     occ_loss = F.smooth_l1_loss(torch.norm(attr, dim = 1, keepdim=True), occ_ratio.unsqueeze(1))
    
#     return attr_loss / count + occ_loss if count > 0 else occ_loss

# v3
# def attribute_loss_v2(gtbox, attr, bg, occ_ratio, temperature=0.1):
#     gtbox = gtbox.detach()
#     attr_loss = 0.0

#     bg = bg[torch.isin(bg, attr,invert=True).all(dim=1)]

#     attr_loss += contrastive_loss(attr, bg, temperature=temperature)

#     occ_loss = F.smooth_l1_loss(torch.norm(attr, dim = 1, keepdim=True), occ_ratio.unsqueeze(1))
    
#     return attr_loss + occ_loss


# #v4
def attribute_loss_v2(attr, bg, temperature=0.1):

    attr_loss = 0.0

    bg = bg[torch.isin(bg, attr,invert=True).all(dim=1)]
    if attr.size(0) > 0 and bg.size(0) > 0:
        attr_loss += contrastive_loss(attr, bg, temperature=temperature)
        attr_loss += contrastive_loss(bg, attr, temperature=temperature)

    return attr_loss + 1e-9

# def attribute_loss_v2(gtbox, attr, occ_ratio, temperature=0.14):
#     gtbox = gtbox.detach()
#     attr_loss = 0.0
#     L = len(gtbox)
#     count = 0
#     for i in range(L):
#             mask = gtbox == gtbox[i]
#             mask[i] = False
#             positive = attr[mask.any(dim=1)]
#             mask[i] = True
#             negative = attr[~mask.any(dim=1)]

#             if positive.size(0) > 0 and negative.size(0) > 0:
#                 count += 1
#                 attr_loss += Cos_triplet_loss(attr[i].unsqueeze(0), positive, negative, temperature=temperature)

#     occ_loss = F.smooth_l1_loss(torch.norm(attr, dim = 1, keepdim=True), occ_ratio)
    
#     return attr_loss / count + occ_loss


def repbox_loss(pbox, gtbox, deta=0.5, pnms=0.1, gtnms=0.1, x1x2y1y2=False):
    repgt_loss = 0.0
    repbox_loss = 0.0
    pbox = pbox.detach()
    gtbox = gtbox.detach()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gtbox_cpu = gtbox.cuda().data.cpu().numpy()
    pgiou = box_iou_v5(pbox, gtbox, x1y1x2y2=x1x2y1y2)
    pgiou = pgiou.cuda().data.cpu().numpy()
    ppiou = box_iou_v5(pbox, pbox, x1y1x2y2=x1x2y1y2)
    ppiou = ppiou.cuda().data.cpu().numpy()
    # t1 = time.time()
    len = pgiou.shape[0]
    for j in range(len):
        for z in range(j, len):
            ppiou[j, z] = 0
            # if int(torch.sum(gtbox[j] == gtbox[z])) == 4:
            # if int(torch.sum(gtbox_cpu[j] == gtbox_cpu[z])) == 4:
            # if int(np.sum(gtbox_numpy[j] == gtbox_numpy[z])) == 4:
            if (gtbox_cpu[j][0]==gtbox_cpu[z][0]) and (gtbox_cpu[j][1]==gtbox_cpu[z][1]) and (gtbox_cpu[j][2]==gtbox_cpu[z][2]) and (gtbox_cpu[j][3]==gtbox_cpu[z][3]):
                pgiou[j, z] = 0
                pgiou[z, j] = 0
                ppiou[z, j] = 0

    pgiou = torch.from_numpy(pgiou).cuda().detach()
    ppiou = torch.from_numpy(ppiou).cuda().detach()

    # repbox
    pp_mask = torch.gt(ppiou, pnms)  # 防止nms为0, 因为如果为0,那么上面的for循环就没有意义了 [N x N] error
    num_pbox = pp_mask.sum()
    if num_pbox > 0:
        repbox_loss = smooth_ln(ppiou, deta)
        repbox_loss = repbox_loss.mean()
    # mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
    # print(mem)
    torch.cuda.empty_cache()

    return repbox_loss