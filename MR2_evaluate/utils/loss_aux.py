import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import bbox_iou, bbox_alpha_iou, box_iou, box_giou, box_diou, box_ciou, xywh2xyxy, Wasserstein
from utils.torch_utils import is_parallel
from utils.loss_rep import repulsion_loss_torch, repbox_loss, IoG, attribute_loss, attribute_loss_v2

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

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

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLossAuxOTA_v2_rep:
    # Compute losses
    def __init__(self, model, visible=True, autobalance=False):
        super(ComputeLossAuxOTA_v2_rep, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

        self.visible = visible
        if self.visible:
            print('Use visible bounding box for auxiliary head training')
        else:
            print('Use fully bounding box for auxiliary head training')
            # print('Use yolov7 label assignment fully bounding box for trianing')
    def __call__(self, p, targets, imgs):  # predictions, targets, model   
        device = targets.device
        # print(p)
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lrepBox, lrepGT = torch.zeros(1, device=device), torch.zeros(1, device=device)
        # v1(good) for CP
        # bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets(p[self.nl:], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
        # v2(bad)
        # bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets(p[:self.nl], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
        # yolov7 assignment method for TJU
        # bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets2(p[:self.nl], targets[:, :6], imgs)

        if self.visible:
            bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets(p[self.nl:], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
        else:
            # independent assigner
            bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets(p[self.nl:], targets[:, :6], imgs)
            # YOLOv7 coarse to fine guided assigner
            # bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets2(p[:self.nl], targets[:, :6], imgs)

        bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.nl], targets[:, :6], imgs)
        pre_gen_gains_aux = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
    

        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_aux = p[i+self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            b_aux, a_aux, gj_aux, gi_aux = bs_aux[i], as_aux_[i], gjs_aux[i], gis_aux[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_aux = torch.zeros_like(pi_aux[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                # print(pi.size(),pbox.size(), selected_tbox.size())
            
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                # if self.hyp['was'] != 0:
                #     nwd = torch.exp(-torch.pow(Wasserstein(pbox.T, selected_tbox, x1y1x2y2=False), 1 / 2)) #Wasserstein loss
                #     lbox += (1-self.hyp['was']) * (1.0 - iou).mean() + self.hyp['was'] * (1.0 - nwd).mean() # iou + wasserstein loss
                # elif self.hyp['was'] == 0:
                #     lbox += (1.0 - iou).mean()  # iou loss
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

                # Repulsion Loss
                dic = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
                    13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [],
                    25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: [], 32: [], 33: [], 34: [], 35: [], 36: [],
                    37: [], 38: [], 39: [], 40: [], 41: [], 42: [], 43: [], 44: [], 45: [], 46: [], 47: [], 48: [], 
                    49: [], 50: [], 51: [], 52: [], 53: [], 54: [], 55: [], 56: [], 57: [], 58: [], 59: [], 60: [], 
                    61: [], 62: [], 63: [], 64: [], 65: [], 66: [], 67: [], 68: [], 69: [], 70: [], 71: [], 72: [], 
                    73: [], 74: [], 75: [], 76: [], 77: [], 78: [], 79: [], 80: [], 81: [], 82: [], 83: [], 84: [], 
                    85: [], 86: [], 87: [], 88: [], 89: [], 90: [], 91: [], 92: [], 93: [], 94: [], 95: [], 96: [], 
                    97: [], 98: [], 99: [], 100: [], 101: [], 102: [], 103: [], 104: [], 105: [], 106: [], 107: [], 
                    108: [], 109: [], 110: [], 111: [], 112: [], 113: [], 114: [], 115: [], 116: [], 117: [], 118: [], 
                    119: [], 120: [], 121: [], 122: [], 123: [], 124: [], 125: [], 126: [], 127: [], 128: [], 129: [],
                    130: [], 131: [], 132: [], 133: [], 134: [], 135: [], 136: [], 137: [], 138: [], 139: [], 140: [], 
                    141: [], 142: [], 143: [], 144: [], 145: [], 146: [], 147: [], 148: [], 149: []}
                for indexs, value in enumerate(b):
                    # print(indexs, value)
                    dic[int(value)].append(indexs)
                # print('dic', dic)
                bts = 0
                deta = self.hyp['deta']
                Rp_nms = self.hyp['Rp_nms']
                _lrepGT = 0.0
                _lrepBox = 0.0
                for id, indexs in dic.items():  # id = batch_name  indexs = target_id
                    if indexs:
                        lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs], deta=deta, pnms=Rp_nms, gtnms=Rp_nms)
                        # lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs])
                        _lrepGT += lrepgt
                        _lrepBox += lrepbox
                        bts += 1
                if bts > 0:
                    _lrepGT /= bts
                    _lrepBox /= bts
                lrepGT += _lrepGT
                lrepBox += _lrepBox

            n_aux = b_aux.shape[0]  # number of targets
            if n_aux:
                ps_aux = pi_aux[b_aux, a_aux, gj_aux, gi_aux]  # prediction subset corresponding to targets
                grid_aux = torch.stack([gi_aux, gj_aux], dim=1)
                pxy_aux = ps_aux[:, :2].sigmoid() * 2. - 0.5
                #pxy_aux = ps_aux[:, :2].sigmoid() * 3. - 1.
                pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors_aux[i]
                pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)  # predicted box
                selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains_aux[i]
                selected_tbox_aux[:, :2] -= grid_aux
                iou_aux = bbox_iou(pbox_aux.T, selected_tbox_aux, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += 0.25 * (1.0 - iou_aux).mean()  # iou loss

                # Objectness
                tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = (1.0 - self.gr) + self.gr * iou_aux.detach().clamp(0).type(tobj_aux.dtype)  # iou ratio

                # Classification
                selected_tcls_aux = targets_aux[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_aux = torch.full_like(ps_aux[:, 5:], self.cn, device=device)  # targets
                    t_aux[range(n_aux), selected_tcls_aux] = self.cp
                    lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)  # BCE

                #  # Repulsion Loss
                # dic = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
                #     13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [],
                #     25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: [], 32: [], 33: [], 34: [], 35: [], 36: [],
                #     37: [], 38: [], 39: [], 40: [], 41: [], 42: [], 43: [], 44: [], 45: [], 46: [], 47: [], 48: [], 
                #     49: [], 50: [], 51: [], 52: [], 53: [], 54: [], 55: [], 56: [], 57: [], 58: [], 59: [], 60: [], 
                #     61: [], 62: [], 63: [], 64: [], 65: [], 66: [], 67: [], 68: [], 69: [], 70: [], 71: [], 72: [], 
                #     73: [], 74: [], 75: [], 76: [], 77: [], 78: [], 79: [], 80: [], 81: [], 82: [], 83: [], 84: [], 
                #     85: [], 86: [], 87: [], 88: [], 89: [], 90: [], 91: [], 92: [], 93: [], 94: [], 95: [], 96: [], 
                #     97: [], 98: [], 99: [], 100: [], 101: [], 102: [], 103: [], 104: [], 105: [], 106: [], 107: [], 
                #     108: [], 109: [], 110: [], 111: [], 112: [], 113: [], 114: [], 115: [], 116: [], 117: [], 118: [], 
                #     119: [], 120: [], 121: [], 122: [], 123: [], 124: [], 125: [], 126: [], 127: [], 128: [], 129: [],
                #     130: [], 131: [], 132: [], 133: [], 134: [], 135: [], 136: [], 137: [], 138: [], 139: [], 140: [], 
                #     141: [], 142: [], 143: [], 144: [], 145: [], 146: [], 147: [], 148: [], 149: []}
                # for indexs, value in enumerate(b_aux):
                #     # print(indexs, value)
                #     dic[int(value)].append(indexs)
                # # print('dic', dic)
                # bts_aux = 0
                # deta = self.hyp['deta']
                # Rp_nms = self.hyp['Rp_nms']
                # _lrepGT_aux = 0.0
                # _lrepBox_aux = 0.0
                # for id, indexs in dic.items():  # id = batch_name  indexs = target_id
                #     if indexs:
                #         lrepgt, lrepbox = repulsion_loss_torch(pbox_aux[indexs], selected_tbox_aux[indexs], deta=deta, pnms=Rp_nms, gtnms=Rp_nms)
                #         # lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs])
                #         _lrepGT_aux += lrepgt
                #         _lrepBox_aux += lrepbox
                #         bts_aux += 1
                # if bts_aux > 0:
                #     _lrepGT_aux /= bts_aux
                #     _lrepBox_aux /= bts_aux
                # lrepGT += _lrepGT_aux * 0.25
                # lrepBox += _lrepBox_aux * 0.25

            obji = self.BCEobj(pi[..., 4], tobj)
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
            lobj += obji * self.balance[i] + 0.25 * obji_aux * self.balance[i] # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        # exit()
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        if self.hyp['alpha'] == 0:
            lrep = self.hyp['beta'] * lrepBox / 1.5
        else:
            lrep = self.hyp['alpha'] * lrepGT / 3.0 + self.hyp['beta'] * lrepBox / 3.0
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lrep
        return loss * bs, torch.cat((lbox, lobj, lcls, lrep, loss)).detach()

    def build_targets(self, p, targets, imgs):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs           

    def build_targets2(self, p, targets, imgs):
        
        indices, anch = self.find_5_positive(p, targets)
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(20, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs              

    def find_5_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 1.0  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch                 
    
    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch


class ComputeLossAuxOTA_v2_rep_Guide:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossAuxOTA_v2_rep_Guide, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))


    def __call__(self, p, targets, imgs):  # predictions, targets, model   
        device = targets.device
        # print(p)
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lrepBox, lrepGT = torch.zeros(1, device=device), torch.zeros(1, device=device)
      
        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets(p[self.nl:], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
        
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.nl], targets[:, :6], imgs, topK=5)
        # bs, as_, gjs, gis, targets, anchors = self.build_targets_guide(p[:self.nl], targets[:, [0, 1, 6, 7, 8, 9]], targets[:, :6], imgs)
        pre_gen_gains_aux = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
    

        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_aux = p[i+self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            b_aux, a_aux, gj_aux, gi_aux = bs_aux[i], as_aux_[i], gjs_aux[i], gis_aux[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_aux = torch.zeros_like(pi_aux[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                # print(pi.size(),pbox.size(), selected_tbox.size())
            
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

                # Repulsion Loss
                dic = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
                    13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [],
                    25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: [], 32: [], 33: [], 34: [], 35: [], 36: [],
                    37: [], 38: [], 39: [], 40: [], 41: [], 42: [], 43: [], 44: [], 45: [], 46: [], 47: [], 48: [], 
                    49: [], 50: [], 51: [], 52: [], 53: [], 54: [], 55: [], 56: [], 57: [], 58: [], 59: [], 60: [], 
                    61: [], 62: [], 63: [], 64: [], 65: [], 66: [], 67: [], 68: [], 69: [], 70: [], 71: [], 72: [], 
                    73: [], 74: [], 75: [], 76: [], 77: [], 78: [], 79: [], 80: [], 81: [], 82: [], 83: [], 84: [], 
                    85: [], 86: [], 87: [], 88: [], 89: [], 90: [], 91: [], 92: [], 93: [], 94: [], 95: [], 96: [], 
                    97: [], 98: [], 99: [], 100: [], 101: [], 102: [], 103: [], 104: [], 105: [], 106: [], 107: [], 
                    108: [], 109: [], 110: [], 111: [], 112: [], 113: [], 114: [], 115: [], 116: [], 117: [], 118: [], 
                    119: [], 120: [], 121: [], 122: [], 123: [], 124: [], 125: [], 126: [], 127: [], 128: [], 129: [],
                    130: [], 131: [], 132: [], 133: [], 134: [], 135: [], 136: [], 137: [], 138: [], 139: [], 140: [], 
                    141: [], 142: [], 143: [], 144: [], 145: [], 146: [], 147: [], 148: [], 149: []}
                for indexs, value in enumerate(b):
                    # print(indexs, value)
                    dic[int(value)].append(indexs)
                # print('dic', dic)
                bts = 0
                deta = self.hyp['deta']
                Rp_nms = self.hyp['Rp_nms']
                _lrepGT = 0.0
                _lrepBox = 0.0
                for id, indexs in dic.items():  # id = batch_name  indexs = target_id
                    if indexs:
                        lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs], deta=deta, pnms=Rp_nms, gtnms=Rp_nms)
                        # lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs])
                        _lrepGT += lrepgt
                        _lrepBox += lrepbox
                        bts += 1
                if bts > 0:
                    _lrepGT /= bts
                    _lrepBox /= bts
                lrepGT += _lrepGT
                lrepBox += _lrepBox

            n_aux = b_aux.shape[0]  # number of targets
            if n_aux:
                ps_aux = pi_aux[b_aux, a_aux, gj_aux, gi_aux]  # prediction subset corresponding to targets
                grid_aux = torch.stack([gi_aux, gj_aux], dim=1)
                pxy_aux = ps_aux[:, :2].sigmoid() * 2. - 0.5
                #pxy_aux = ps_aux[:, :2].sigmoid() * 3. - 1.
                pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors_aux[i]
                pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)  # predicted box
                selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains_aux[i]
                selected_tbox_aux[:, :2] -= grid_aux
                iou_aux = bbox_iou(pbox_aux.T, selected_tbox_aux, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += 0.25 * (1.0 - iou_aux).mean()  # iou loss

                # Objectness
                tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = (1.0 - self.gr) + self.gr * iou_aux.detach().clamp(0).type(tobj_aux.dtype)  # iou ratio

                # Classification
                selected_tcls_aux = targets_aux[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_aux = torch.full_like(ps_aux[:, 5:], self.cn, device=device)  # targets
                    t_aux[range(n_aux), selected_tcls_aux] = self.cp
                    lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)  # BCE


            obji = self.BCEobj(pi[..., 4], tobj)
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
            lobj += obji * self.balance[i] + 0.25 * obji_aux * self.balance[i] # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        # exit()
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        if self.hyp['alpha'] == 0:
            lrep = self.hyp['beta'] * lrepBox / 1.5
        else:
            lrep = self.hyp['alpha'] * lrepGT / 3.0 + self.hyp['beta'] * lrepBox / 3.0
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lrep
        return loss * bs, torch.cat((lbox, lobj, lcls, lrep, loss)).detach()

    def build_targets(self, p, targets, imgs, topK=10):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(topK, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs           
        

    def build_targets_guide(self, p, targets_aux, targets, imgs):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets_aux)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets_aux[b_idx]
            full_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            full_target = full_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(full_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs   
  
    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch
    
class ComputeLossAuxOTA_v2_rep_triple:
    # Compute losses
    def __init__(self, model, visible=True, autobalance=False):
        super(ComputeLossAuxOTA_v2_rep_triple, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

        self.visible = visible
        if self.visible:
            print('Use visible bounding box for auxiliary head training')
        else:
            print('Use fully bounding box for auxiliary head training')
            # print('Use yolov7 label assignment fully bounding box for trianing')
    def __call__(self, p, targets, imgs):  # predictions, targets, model   
        device = targets.device
        # print(p)
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lrepBox, lrepGT = torch.zeros(1, device=device), torch.zeros(1, device=device)
        # v1(good) for CP
        # bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets(p[self.nl:], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
        # v2(bad)
        # bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets(p[:self.nl], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
        # yolov7 assignment method for TJU
        # bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets2(p[:self.nl], targets[:, :6], imgs)

        if self.visible:
            bs_aux_1, as_aux_1, gjs_aux_1, gis_aux_1, targets_aux_1, anchors_aux_1 = self.build_targets(p[self.nl:2*self.nl], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
            bs_aux_2, as_aux_2, gjs_aux_2, gis_aux_2, targets_aux_2, anchors_aux_2 = self.build_targets(p[2*self.nl:], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
        else:
            # independent assigner
            print('Current only support visible box for triple anchor based head')
            exit()
            # bs_aux_1, as_aux_1, gjs_aux_1, gis_aux_1, targets_aux_1, anchors_aux_1 = self.build_targets(p[self.nl:], targets[:, :6], imgs)

        bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.nl], targets[:, :6], imgs)
        pre_gen_gains_aux = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
    

        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_aux1 = p[i+self.nl]
            pi_aux2 = p[i+2*self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            b_aux1, a_aux1, gj_aux1, gi_aux1 = bs_aux_1[i], as_aux_1[i], gjs_aux_1[i], gis_aux_1[i]  # image, anchor, gridy, gridx
            b_aux2, a_aux2, gj_aux2, gi_aux2 = bs_aux_2[i], as_aux_2[i], gjs_aux_2[i], gis_aux_2[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_aux1 = torch.zeros_like(pi_aux1[..., 0], device=device)  # target obj
            tobj_aux2 = torch.zeros_like(pi_aux2[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                # if self.hyp['was'] != 0:
                #     nwd = torch.exp(-torch.pow(Wasserstein(pbox.T, selected_tbox, x1y1x2y2=False), 1 / 2)) #Wasserstein loss
                #     lbox += (1-self.hyp['was']) * (1.0 - iou).mean() + self.hyp['was'] * (1.0 - nwd).mean() # iou + wasserstein loss
                # elif self.hyp['was'] == 0:
                #     lbox += (1.0 - iou).mean()  # iou loss
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

                # Repulsion Loss
                dic = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
                    13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [],
                    25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: [], 32: [], 33: [], 34: [], 35: [], 36: [],
                    37: [], 38: [], 39: [], 40: [], 41: [], 42: [], 43: [], 44: [], 45: [], 46: [], 47: [], 48: [], 
                    49: [], 50: [], 51: [], 52: [], 53: [], 54: [], 55: [], 56: [], 57: [], 58: [], 59: [], 60: [], 
                    61: [], 62: [], 63: [], 64: [], 65: [], 66: [], 67: [], 68: [], 69: [], 70: [], 71: [], 72: [], 
                    73: [], 74: [], 75: [], 76: [], 77: [], 78: [], 79: [], 80: [], 81: [], 82: [], 83: [], 84: [], 
                    85: [], 86: [], 87: [], 88: [], 89: [], 90: [], 91: [], 92: [], 93: [], 94: [], 95: [], 96: [], 
                    97: [], 98: [], 99: [], 100: [], 101: [], 102: [], 103: [], 104: [], 105: [], 106: [], 107: [], 
                    108: [], 109: [], 110: [], 111: [], 112: [], 113: [], 114: [], 115: [], 116: [], 117: [], 118: [], 
                    119: [], 120: [], 121: [], 122: [], 123: [], 124: [], 125: [], 126: [], 127: [], 128: [], 129: [],
                    130: [], 131: [], 132: [], 133: [], 134: [], 135: [], 136: [], 137: [], 138: [], 139: [], 140: [], 
                    141: [], 142: [], 143: [], 144: [], 145: [], 146: [], 147: [], 148: [], 149: []}
                for indexs, value in enumerate(b):
                    # print(indexs, value)
                    dic[int(value)].append(indexs)
                # print('dic', dic)
                bts = 0
                deta = self.hyp['deta']
                Rp_nms = self.hyp['Rp_nms']
                _lrepGT = 0.0
                _lrepBox = 0.0
                for id, indexs in dic.items():  # id = batch_name  indexs = target_id
                    if indexs:
                        lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs], deta=deta, pnms=Rp_nms, gtnms=Rp_nms)
                        # lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs])
                        _lrepGT += lrepgt
                        _lrepBox += lrepbox
                        bts += 1
                if bts > 0:
                    _lrepGT /= bts
                    _lrepBox /= bts
                lrepGT += _lrepGT
                lrepBox += _lrepBox

            n_aux_1 = b_aux1.shape[0]  # number of targets
            if n_aux_1:
                ps_aux = pi_aux1[b_aux1, a_aux1, gj_aux1, gi_aux1]  # prediction subset corresponding to targets
                grid_aux = torch.stack([gi_aux1, gj_aux1], dim=1)
                pxy_aux = ps_aux[:, :2].sigmoid() * 2. - 0.5
                #pxy_aux = ps_aux[:, :2].sigmoid() * 3. - 1.
                pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors_aux_1[i]
                pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)  # predicted box
                selected_tbox_aux = targets_aux_1[i][:, 2:6] * pre_gen_gains_aux[i]
                selected_tbox_aux[:, :2] -= grid_aux
                iou_aux = bbox_iou(pbox_aux.T, selected_tbox_aux, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += 0.125 * (1.0 - iou_aux).mean()  # iou loss

                # Objectness
                tobj_aux1[b_aux1, a_aux1, gj_aux1, gi_aux1] = (1.0 - self.gr) + self.gr * iou_aux.detach().clamp(0).type(tobj_aux1.dtype)  # iou ratio

                # Classification
                selected_tcls_aux = targets_aux_1[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_aux = torch.full_like(ps_aux[:, 5:], self.cn, device=device)  # targets
                    t_aux[range(n_aux_1), selected_tcls_aux] = self.cp
                    lcls += 0.125 * self.BCEcls(ps_aux[:, 5:], t_aux)  # BCE

            n_aux_2 = b_aux2.shape[0]  # number of targets
            if n_aux_2:
                ps_aux = pi_aux2[b_aux2, a_aux2, gj_aux2, gi_aux2]  # prediction subset corresponding to targets
                grid_aux = torch.stack([gi_aux2, gj_aux2], dim=1)
                pxy_aux = ps_aux[:, :2].sigmoid() * 2. - 0.5
                #pxy_aux = ps_aux[:, :2].sigmoid() * 3. - 1.
                pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors_aux_2[i]
                pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)  # predicted box
                selected_tbox_aux = targets_aux_2[i][:, 2:6] * pre_gen_gains_aux[i]
                selected_tbox_aux[:, :2] -= grid_aux
                iou_aux = bbox_iou(pbox_aux.T, selected_tbox_aux, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += 0.25 * (1.0 - iou_aux).mean()  # iou loss

                # Objectness
                tobj_aux2[b_aux2, a_aux2, gj_aux2, gi_aux2] = (1.0 - self.gr) + self.gr * iou_aux.detach().clamp(0).type(tobj_aux2.dtype)  # iou ratio

                # Classification
                selected_tcls_aux = targets_aux_2[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_aux = torch.full_like(ps_aux[:, 5:], self.cn, device=device)  # targets
                    t_aux[range(n_aux_1), selected_tcls_aux] = self.cp
                    lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)  # BCE
                    
            obji = self.BCEobj(pi[..., 4], tobj)
            obji_aux1 = self.BCEobj(pi_aux1[..., 4], tobj_aux1)
            obji_aux2 = self.BCEobj(pi_aux2[..., 4], tobj_aux2)
            lobj += obji * self.balance[i] + 0.125 * obji_aux1 * self.balance[i] + 0.25 * obji_aux2 # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        if self.hyp['alpha'] == 0:
            lrep = self.hyp['beta'] * lrepBox / 1.5
        else:
            lrep = self.hyp['alpha'] * lrepGT / 3.0 + self.hyp['beta'] * lrepBox / 3.0
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lrep
        return loss * bs, torch.cat((lbox, lobj, lcls, lrep, loss)).detach()

    def build_targets(self, p, targets, imgs):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs                     

    def find_5_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 1.0  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch                 
    
    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch


class ComputeLossAuxOTA_v2_rep_den:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossAuxOTA_v2_rep_den, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs, density):  # predictions, targets, model   
        device = targets.device

        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lrepBox, lrepGT = torch.zeros(1, device=device), torch.zeros(1, device=device)
        lattr = torch.zeros(1, device=device)

        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets(p[self.nl:], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.nl], targets[:, :6], imgs)
        pre_gen_gains_aux = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
    

        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_aux = p[i+self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            b_aux, a_aux, gj_aux, gi_aux = bs_aux[i], as_aux_[i], gjs_aux[i], gis_aux[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_aux = torch.zeros_like(pi_aux[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                
                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lattr += attribute_loss(selected_tbox, density[i][b, a, gj, gi])
                
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

                # Repulsion Loss
                dic = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
                       13: [], 14: [], 15: []}
                for indexs, value in enumerate(b):
                    # print(indexs, value)
                    dic[int(value)].append(indexs)
                # print('dic', dic)
                bts = 0
                deta = self.hyp['deta']
                Rp_nms = self.hyp['Rp_nms']
                _lrepGT = 0.0
                _lrepBox = 0.0
                for id, indexs in dic.items():  # id = batch_name  indexs = target_id
                    if indexs:
                        lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs], deta=deta, pnms=Rp_nms, gtnms=Rp_nms)
                        # lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs])
                        _lrepGT += lrepgt
                        _lrepBox += lrepbox
                        bts += 1
                if bts > 0:
                    _lrepGT /= bts
                    _lrepBox /= bts
                lrepGT += _lrepGT
                lrepBox += _lrepBox

            n_aux = b_aux.shape[0]  # number of targets
            if n_aux:
                ps_aux = pi_aux[b_aux, a_aux, gj_aux, gi_aux]  # prediction subset corresponding to targets
                grid_aux = torch.stack([gi_aux, gj_aux], dim=1)
                pxy_aux = ps_aux[:, :2].sigmoid() * 2. - 0.5
                #pxy_aux = ps_aux[:, :2].sigmoid() * 3. - 1.
                pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors_aux[i]
                pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)  # predicted box
                selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains_aux[i]
                selected_tbox_aux[:, :2] -= grid_aux
                iou_aux = bbox_iou(pbox_aux.T, selected_tbox_aux, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += 0.25 * (1.0 - iou_aux).mean()  # iou loss

                # Objectness
                tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = (1.0 - self.gr) + self.gr * iou_aux.detach().clamp(0).type(tobj_aux.dtype)  # iou ratio

                # Classification
                selected_tcls_aux = targets_aux[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_aux = torch.full_like(ps_aux[:, 5:], self.cn, device=device)  # targets
                    t_aux[range(n_aux), selected_tcls_aux] = self.cp
                    lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
            lobj += obji * self.balance[i] + 0.25 * obji_aux * self.balance[i] # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lattr *= self.hyp['attr']
        if self.hyp['alpha'] == 0:
            lrep = self.hyp['beta'] * lrepBox / 1.5
        else:
            lrep = self.hyp['alpha'] * lrepGT / 3.0 + self.hyp['beta'] * lrepBox / 3.0
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lrep + lattr
        return loss * bs, torch.cat((lbox, lobj, lattr, lrep, loss)).detach()

    def build_targets(self, p, targets, imgs):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs           

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch
  

class ComputeLossAuxOTA_v2_rep_den_v2:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossAuxOTA_v2_rep_den_v2, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs, density):  # predictions, targets, model   
        device = targets.device

        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lrepBox, lrepGT = torch.zeros(1, device=device), torch.zeros(1, device=device)
        lattr = torch.zeros(1, device=device)

        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets(p[self.nl:], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
        # v2 v3
        # bs, as_, gjs, gis, targets, anchors, occ_ratio = self.build_targets_occ_ratio(p[:self.nl], targets[:, :6], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
        # v4
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.nl], targets[:, :6], imgs)
        pre_gen_gains_aux = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
    

        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_aux = p[i+self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            b_aux, a_aux, gj_aux, gi_aux = bs_aux[i], as_aux_[i], gjs_aux[i], gis_aux[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_aux = torch.zeros_like(pi_aux[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                # lattr += attribute_loss(selected_tbox, density[i][b, a, gj, gi])
                n_b = torch.randint(0, pi.size(0), (len(selected_tbox), ))
                n_a = torch.randint(0, pi.size(1), (len(selected_tbox), ))
                n_gj = torch.randint(0, pi.size(2), (len(selected_tbox), ))
                n_gi = torch.randint(0, pi.size(3), (len(selected_tbox), ))
                # v2 v3
                # lattr += attribute_loss_v2(selected_tbox, density[i][b, a, gj, gi], density[i][n_b, n_a, n_gj, n_gi], occ_ratio[i])
                # v4
                lattr += attribute_loss_v2(density[i][b, a, gj, gi], density[i][n_b, n_a, n_gj, n_gi])
                
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

                # Repulsion Loss
                dic = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
                       13: [], 14: [], 15: []}
                for indexs, value in enumerate(b):
                    # print(indexs, value)
                    dic[int(value)].append(indexs)
                # print('dic', dic)
                bts = 0
                deta = self.hyp['deta']
                Rp_nms = self.hyp['Rp_nms']
                _lrepGT = 0.0
                _lrepBox = 0.0
                for id, indexs in dic.items():  # id = batch_name  indexs = target_id
                    if indexs:
                        lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs], deta=deta, pnms=Rp_nms, gtnms=Rp_nms)
                        # lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs])
                        _lrepGT += lrepgt
                        _lrepBox += lrepbox
                        bts += 1
                if bts > 0:
                    _lrepGT /= bts
                    _lrepBox /= bts
                lrepGT += _lrepGT
                lrepBox += _lrepBox

            n_aux = b_aux.shape[0]  # number of targets
            if n_aux:
                ps_aux = pi_aux[b_aux, a_aux, gj_aux, gi_aux]  # prediction subset corresponding to targets
                grid_aux = torch.stack([gi_aux, gj_aux], dim=1)
                pxy_aux = ps_aux[:, :2].sigmoid() * 2. - 0.5
                #pxy_aux = ps_aux[:, :2].sigmoid() * 3. - 1.
                pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors_aux[i]
                pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)  # predicted box
                selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains_aux[i]
                selected_tbox_aux[:, :2] -= grid_aux
                iou_aux = bbox_iou(pbox_aux.T, selected_tbox_aux, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += 0.25 * (1.0 - iou_aux).mean()  # iou loss

                # Objectness
                tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = (1.0 - self.gr) + self.gr * iou_aux.detach().clamp(0).type(tobj_aux.dtype)  # iou ratio

                # Classification
                selected_tcls_aux = targets_aux[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_aux = torch.full_like(ps_aux[:, 5:], self.cn, device=device)  # targets
                    t_aux[range(n_aux), selected_tcls_aux] = self.cp
                    lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
            lobj += obji * self.balance[i] + 0.25 * obji_aux * self.balance[i] # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lattr *= self.hyp['attr']
        # lattr *= 0.001
        if self.hyp['alpha'] == 0:
            lrep = self.hyp['beta'] * lrepBox / 1.5
        else:
            lrep = self.hyp['alpha'] * lrepGT / 3.0 + self.hyp['beta'] * lrepBox / 3.0
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lrep + lattr
        return loss * bs, torch.cat((lbox, lobj, lattr, lrep, loss)).detach()

    def build_targets_occ_ratio(self, p, targets, targets_occ, imgs):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]

        matching_weight =  [[] for pp in p]

        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            this_occ = targets_occ[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            txywh_occ = this_occ[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy_occ = xywh2xyxy(txywh_occ)
            vis_IOG = IoG(txyxy, txyxy_occ)
            # print(txyxy)
            # print(txyxy_occ)
            # print(vis_IOG)
            # print(2-vis_IOG)
            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
            vis_IOG = vis_IOG[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_weight[i].append(vis_IOG[layer_idx]) # visiable
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_weight[i] = torch.cat(matching_weight[i], dim=0) # visiable
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_weight[i] = torch.tensor([], device='cuda:0', dtype=torch.int64) # visiable
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        # for i in range(len(matching_targets)):
        #     print((matching_targets[i]))
        #     print((matching_weight[i]))
        # exit()
        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs, matching_weight
    
    def build_targets(self, p, targets, imgs):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs           

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch
  

class ComputeLossAuxOTA_v2:
    # Compute losses
    def __init__(self, model, visible=True, autobalance=False):
        super(ComputeLossAuxOTA_v2, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

        self.visible = visible
        if self.visible:
            print('Use visible bounding box for auxiliary head training')
        else:
            print('Use fully bounding box for auxiliary head training')

    def __call__(self, p, targets, imgs):  # predictions, targets, model   
        device = targets.device
        # print(p)
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        if self.visible:
            bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets(p[self.nl:], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
        else:
            # independent assigner
            bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets(p[self.nl:], targets[:, :6], imgs)
            # YOLOv7 coarse to fine guided assigner
            # bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets2(p[:self.nl], targets[:, :6], imgs)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.nl], targets[:, :6], imgs)
        pre_gen_gains_aux = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 

        


        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_aux = p[i+self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            b_aux, a_aux, gj_aux, gi_aux = bs_aux[i], as_aux_[i], gjs_aux[i], gis_aux[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_aux = torch.zeros_like(pi_aux[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            
            n_aux = b_aux.shape[0]  # number of targets
            if n_aux:
                ps_aux = pi_aux[b_aux, a_aux, gj_aux, gi_aux]  # prediction subset corresponding to targets
                grid_aux = torch.stack([gi_aux, gj_aux], dim=1)
                pxy_aux = ps_aux[:, :2].sigmoid() * 2. - 0.5
                #pxy_aux = ps_aux[:, :2].sigmoid() * 3. - 1.
                pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors_aux[i]
                pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)  # predicted box
                selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains_aux[i]
                selected_tbox_aux[:, :2] -= grid_aux
                iou_aux = bbox_iou(pbox_aux.T, selected_tbox_aux, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += 0.25 * (1.0 - iou_aux).mean()  # iou loss

                # Objectness
                tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = (1.0 - self.gr) + self.gr * iou_aux.detach().clamp(0).type(tobj_aux.dtype)  # iou ratio

                # Classification
                selected_tcls_aux = targets_aux[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_aux = torch.full_like(ps_aux[:, 5:], self.cn, device=device)  # targets
                    t_aux[range(n_aux), selected_tcls_aux] = self.cp
                    lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
            lobj += obji * self.balance[i] + 0.25 * obji_aux * self.balance[i] # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets, imgs):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs           

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch

class ComputeLossAuxOTA_v2_rep_map:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossAuxOTA_v2_rep_map, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        
        BCEmap = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['map_pw']], device=device))
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj, BCEmap = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCEmap, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.BCEmap, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, BCEmap, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, pm, maps, imgs):  # predictions, targets, model   
        device = targets.device
        # print(p)
        lcls, lbox, lobj, lmap = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lrepBox, lrepGT = torch.zeros(1, device=device), torch.zeros(1, device=device)

        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets(p[self.nl:], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.nl], targets[:, :6], imgs)
        pre_gen_gains_aux = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
    
        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_aux = p[i+self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            b_aux, a_aux, gj_aux, gi_aux = bs_aux[i], as_aux_[i], gjs_aux[i], gis_aux[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_aux = torch.zeros_like(pi_aux[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                # if self.hyp['was'] != 0:
                #     nwd = torch.exp(-torch.pow(Wasserstein(pbox.T, selected_tbox, x1y1x2y2=False), 1 / 2)) #Wasserstein loss
                #     lbox += (1-self.hyp['was']) * (1.0 - iou).mean() + self.hyp['was'] * (1.0 - nwd).mean() # iou + wasserstein loss
                # elif self.hyp['was'] == 0:
                #     lbox += (1.0 - iou).mean()  # iou loss
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

                # Repulsion Loss
                dic = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
                       13: [], 14: [], 15: []}
                for indexs, value in enumerate(b):
                    # print(indexs, value)
                    dic[int(value)].append(indexs)
                # print('dic', dic)
                bts = 0
                deta = self.hyp['deta']
                Rp_nms = self.hyp['Rp_nms']
                _lrepGT = 0.0
                _lrepBox = 0.0
                for id, indexs in dic.items():  # id = batch_name  indexs = target_id
                    if indexs:
                        lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs], deta=deta, pnms=Rp_nms, gtnms=Rp_nms)
                        # lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs])
                        _lrepGT += lrepgt
                        _lrepBox += lrepbox
                        bts += 1
                if bts > 0:
                    _lrepGT /= bts
                    _lrepBox /= bts
                lrepGT += _lrepGT
                lrepBox += _lrepBox

            n_aux = b_aux.shape[0]  # number of targets
            if n_aux:
                ps_aux = pi_aux[b_aux, a_aux, gj_aux, gi_aux]  # prediction subset corresponding to targets
                grid_aux = torch.stack([gi_aux, gj_aux], dim=1)
                pxy_aux = ps_aux[:, :2].sigmoid() * 2. - 0.5
                #pxy_aux = ps_aux[:, :2].sigmoid() * 3. - 1.
                pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors_aux[i]
                pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)  # predicted box
                selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains_aux[i]
                selected_tbox_aux[:, :2] -= grid_aux
                iou_aux = bbox_iou(pbox_aux.T, selected_tbox_aux, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += 0.25 * (1.0 - iou_aux).mean()  # iou loss

                # Objectness
                tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = (1.0 - self.gr) + self.gr * iou_aux.detach().clamp(0).type(tobj_aux.dtype)  # iou ratio

                # Classification
                selected_tcls_aux = targets_aux[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_aux = torch.full_like(ps_aux[:, 5:], self.cn, device=device)  # targets
                    t_aux[range(n_aux), selected_tcls_aux] = self.cp
                    lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
            lobj += obji * self.balance[i] + 0.25 * obji_aux * self.balance[i] # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lmap += self.BCEmap(pm, maps)

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lmap *= self.hyp['map']
        lrep = self.hyp['alpha'] * lrepGT / 3.0 + self.hyp['beta'] * lrepBox / 3.0

        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lrep + lmap
        return loss * bs, torch.cat((lbox, lobj, lcls, lrep, lmap, loss)).detach()

    def build_targets(self, p, targets, imgs):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs           

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch
    

class ComputeLossAuxOTA_v3_rep:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossAuxOTA_v3_rep, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs):  # predictions, targets, model   
        device = targets.device
        # print(p)
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lrepBox, lrepGT = torch.zeros(1, device=device), torch.zeros(1, device=device)
    
        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets(p[self.nl:], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
        bs, as_, gjs, gis, targets, anchors, vis_wgt= self.build_targets_wgt(p[:self.nl], targets[:, :6], targets[:, [0, 1, 6, 7, 8, 9]], imgs)
        pre_gen_gains_aux = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
    
        wgt_max = 1.5
        wgt_ratio = (wgt_max - 1) / 1
        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_aux = p[i+self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            b_aux, a_aux, gj_aux, gi_aux = bs_aux[i], as_aux_[i], gjs_aux[i], gis_aux[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_aux = torch.zeros_like(pi_aux[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)

                lbox += (((vis_wgt[i]-1) * wgt_ratio + 1) * (1.0 - iou)).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

                # Repulsion Loss
                dic = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
                       13: [], 14: [], 15: []}
                for indexs, value in enumerate(b):
                    # print(indexs, value)
                    dic[int(value)].append(indexs)
                # print('dic', dic)
                bts = 0
                deta = self.hyp['deta']
                Rp_nms = self.hyp['Rp_nms']
                _lrepGT = 0.0
                _lrepBox = 0.0
                for id, indexs in dic.items():  # id = batch_name  indexs = target_id
                    if indexs:
                        lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs], deta=deta, pnms=Rp_nms, gtnms=Rp_nms)
                        # lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], selected_tbox[indexs])
                        _lrepGT += lrepgt
                        _lrepBox += lrepbox
                        bts += 1
                if bts > 0:
                    _lrepGT /= bts
                    _lrepBox /= bts
                lrepGT += _lrepGT
                lrepBox += _lrepBox

            n_aux = b_aux.shape[0]  # number of targets
            if n_aux:
                ps_aux = pi_aux[b_aux, a_aux, gj_aux, gi_aux]  # prediction subset corresponding to targets
                grid_aux = torch.stack([gi_aux, gj_aux], dim=1)
                pxy_aux = ps_aux[:, :2].sigmoid() * 2. - 0.5
                #pxy_aux = ps_aux[:, :2].sigmoid() * 3. - 1.
                pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors_aux[i]
                pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)  # predicted box
                selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains_aux[i]
                selected_tbox_aux[:, :2] -= grid_aux
                iou_aux = bbox_iou(pbox_aux.T, selected_tbox_aux, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += 0.25 * (1.0 - iou_aux).mean()  # iou loss

                # Objectness
                tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = (1.0 - self.gr) + self.gr * iou_aux.detach().clamp(0).type(tobj_aux.dtype)  # iou ratio

                # Classification
                selected_tcls_aux = targets_aux[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_aux = torch.full_like(ps_aux[:, 5:], self.cn, device=device)  # targets
                    t_aux[range(n_aux), selected_tcls_aux] = self.cp
                    lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
            lobj += obji * self.balance[i] + 0.25 * obji_aux * self.balance[i] # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        if self.hyp['alpha'] == 0:
            lrep = self.hyp['beta'] * lrepBox / 1.5
        else:
            lrep = self.hyp['alpha'] * lrepGT / 3.0 + self.hyp['beta'] * lrepBox / 3.0
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lrep
        return loss * bs, torch.cat((lbox, lobj, lcls, lrep, loss)).detach()

    def build_targets(self, p, targets, imgs):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs     


    def build_targets_wgt(self, p, targets, targets_occ, imgs):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]

        matching_weight =  [[] for pp in p]

        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            this_occ = targets_occ[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            txywh_occ = this_occ[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy_occ = xywh2xyxy(txywh_occ)
            vis_IOG = 2 - IoG(txyxy, txyxy_occ)
            # print(txyxy)
            # print(txyxy_occ)
            # print(vis_IOG)
            # print(2-vis_IOG)
            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
            vis_IOG = vis_IOG[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_weight[i].append(vis_IOG[layer_idx]) # visiable
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_weight[i] = torch.cat(matching_weight[i], dim=0) # visiable
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_weight[i] = torch.tensor([], device='cuda:0', dtype=torch.int64) # visiable
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        # for i in range(len(matching_targets)):
        #     print((matching_targets[i]))
        #     print((matching_weight[i]))
        # exit()
        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs, matching_weight      

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch