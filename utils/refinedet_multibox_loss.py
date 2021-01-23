import torch
import torch.nn as nn
from torch.nn import functional
from utils.config import coco as cfg
from utils.box_utils import LogSumExp, RefineMatch


class RefineDetMultiBoxLoss(nn.Module):

    def __init__(self, num_classes: int, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, theta=0.01, use_arm=False):
        super(RefineDetMultiBoxLoss, self).__init__()
        self.use_gpu: bool = use_gpu
        self.num_classes: int = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.theta: float = theta
        self.use_arm = use_arm

    def forward(self, predictions, targets):
        arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, priors = predictions
        if self.use_arm:
            loc_data, conf_data = odm_loc_data, odm_conf_data
        else:
            loc_data, conf_data = arm_loc_data, arm_conf_data
        num: int = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors: int = priors.size(0)

        # match priors (default boxes) and ground truth boxes
        loc_t: torch.Tensor = torch.Tensor(num, num_priors, 4)
        conf_t: torch.LongTensor = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            if self.num_classes == 2:
                labels = labels >= 0
            defaults = priors.data
            if self.use_arm:
                RefineMatch(self.threshold, truths, defaults, self.variance,
                            labels, loc_t, conf_t, idx, arm_loc_data[idx].data)()
            else:
                RefineMatch(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)()
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t.requires_grad = False
        conf_t.requires_grad = False

        if self.use_arm:
            p_tmp = functional.softmax(arm_conf_data, 2)
            arm_conf_tmp = p_tmp[:, :, 1]
            object_score_index: torch.Tensor = arm_conf_tmp <= self.theta
            pos: torch.Tensor = torch.as_tensor(conf_t > 0)
            pos[object_score_index.detach()] = 0
        else:
            pos: torch.Tensor = torch.as_tensor(conf_t > 0)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = functional.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = LogSumExp(batch_conf)() - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size()[1]-1)
        neg: torch.Tensor = torch.as_tensor(idx_rank < num_neg.expand_as(idx_rank))

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = functional.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        total_cnt = num_pos.detach().sum().float()
        loss_l /= total_cnt
        loss_c /= total_cnt
        return loss_l, loss_c
