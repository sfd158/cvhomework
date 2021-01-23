import torch
from typing import Optional, List


class PointForm:
    def __init__(self, boxes: torch.Tensor):
        self.boxes = boxes
        self.res: Optional[torch.Tensor] = None

    def __call__(self) -> torch.Tensor:
        self.res = torch.cat((self.boxes[:, :2] - self.boxes[:, 2:] / 2, self.boxes[:, :2] + self.boxes[:, 2:] / 2), 1)
        return self.res


class CenterSize:
    def __init__(self, boxes: torch.Tensor):
        self.boxes = boxes.cuda()
        self.res: Optional[torch.Tensor] = None

    def __call__(self) -> torch.Tensor:
        self.res = torch.cat([(self.boxes[:, 2:] + self.boxes[:, :2])/2, self.boxes[:, 2:] - self.boxes[:, :2]], 1)
        return self.res


class Intersect:
    def __init__(self, box_a: torch.Tensor, box_b: torch.Tensor):
        self.box_a: torch.Tensor = box_a.cuda()
        self.box_b: torch.Tensor = box_b.cuda()
        self.res: Optional[torch.Tensor] = None

    def __call__(self) -> torch.Tensor:
        a: int = self.box_a.size()[0]
        b: int = self.box_b.size()[0]
        max_xy: torch.Tensor = torch.min(self.box_a[:, 2:].unsqueeze(1).expand(a, b, 2),
                                         self.box_b[:, 2:].unsqueeze(0).expand(a, b, 2))
        min_xy: torch.Tensor = torch.max(self.box_a[:, :2].unsqueeze(1).expand(a, b, 2),
                                         self.box_b[:, :2].unsqueeze(0).expand(a, b, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        self.res = inter[:, :, 0] * inter[:, :, 1]
        return self.res


class Jaccard:
    def __init__(self, box_a: torch.Tensor, box_b: torch.Tensor):
        self.box_a: torch.Tensor = box_a.cuda()
        self.box_b: torch.Tensor = box_b.cuda()
        self.res: Optional[torch.Tensor] = None

    def __call__(self) -> torch.Tensor:
        inter: torch.Tensor = Intersect(self.box_a, self.box_b)()
        area_a = ((self.box_a[:, 2] - self.box_a[:, 0]) *
                  (self.box_a[:, 3] - self.box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((self.box_b[:, 2] - self.box_b[:, 0]) *
                  (self.box_b[:, 3] - self.box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        res = inter / union  # [A,B]
        return res


class Match:
    def __init__(self, threshold, truths: torch.Tensor, priors: torch.Tensor, variances: List[float],
                 labels, loc_t, conf_t, idx):
        self.threshold = threshold
        self.truths: torch.Tensor = truths
        self.priors: torch.Tensor = priors
        self.variances = variances
        self.labels = labels
        self.loc_t = loc_t
        self.conf_t = conf_t
        self.idx = idx

    def __call__(self):
        overlaps: torch.Tensor = Jaccard(self.truths, PointForm(self.priors)())()
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
        best_truth_idx.squeeze_(0)
        best_truth_overlap.squeeze_(0)
        best_prior_idx.squeeze_(1)
        best_prior_overlap.squeeze_(1)
        best_truth_overlap.index_fill_(0, best_prior_idx, 2)
        # ensure every gt matches with its prior of max overlap
        for j in range(best_prior_idx.size(0)):
            best_truth_idx[best_prior_idx[j]] = j
        matches: torch.Tensor = self.truths[best_truth_idx]  # Shape: [num_priors,4]
        conf = self.labels[best_truth_idx] + 1  # Shape: [num_priors]
        conf[best_truth_overlap < self.threshold] = 0  # label as background
        loc = Encode(matches, self.priors, self.variances)()
        self.loc_t[self.idx] = loc  # [num_priors,4] encoded offsets to learn
        self.conf_t[self.idx] = conf  # [num_priors] top class label for each prior


class RefineMatch:
    def __init__(self, threshold, truths: torch.Tensor, priors, variances, labels, loc_t, conf_t, idx, arm_loc=None):
        self.threshold = threshold
        self.truths: torch.Tensor = truths
        self.priors = priors
        self.variances = variances
        self.labels = labels
        self.loc_t = loc_t
        self.conf_t = conf_t
        self.idx = idx
        self.arm_loc: torch.Tensor = arm_loc

    def __call__(self):
        decode_arm: Optional[torch.Tensor] = None
        if self.arm_loc is None:
            overlaps: torch.Tensor = Jaccard(self.truths, PointForm(self.priors)())()
        else:
            decode_arm: torch.Tensor = Decode(self.arm_loc, priors=self.priors, variances=self.variances)()
            overlaps: torch.Tensor = Jaccard(self.truths, decode_arm)()

        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
        # [1,num_priors] best ground truth for each prior
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
        best_truth_idx.squeeze_(0)
        best_truth_overlap.squeeze_(0)
        best_prior_idx.squeeze_(1)
        best_prior_overlap.squeeze_(1)
        best_truth_overlap.index_fill_(0, best_prior_idx, 2)

        for j in range(best_prior_idx.size(0)):
            best_truth_idx[best_prior_idx[j]] = j
        matches = self.truths[best_truth_idx]  # Shape: [num_priors,4]
        if self.arm_loc is None:
            conf = self.labels[best_truth_idx]  # Shape: [num_priors]
            loc = Encode(matches, self.priors, self.variances)()
        else:
            conf = self.labels[best_truth_idx] + 1  # Shape: [num_priors]
            loc = Encode(matches, CenterSize(decode_arm)(), self.variances)()
        conf[best_truth_overlap < self.threshold] = 0  # label as background
        self.loc_t[self.idx] = loc  # [num_priors,4] encoded offsets to learn
        self.conf_t[self.idx] = conf  # [num_priors] top class label for each prior


class Encode:
    def __init__(self, matched, priors, variances):
        self.matched = matched.cuda()
        self.priors = priors.cuda()
        self.variances = variances
        self.res: Optional[torch.Tensor] = None

    def __call__(self) -> torch.Tensor:
        g_cxcy: torch.Tensor = (self.matched[:, :2] + self.matched[:, 2:]) / 2 - self.priors[:, :2]
        g_cxcy /= (self.variances[0] * self.priors[:, 2:])
        g_wh: torch.Tensor = (self.matched[:, 2:] - self.matched[:, :2]) / self.priors[:, 2:]
        g_wh: torch.Tensor = torch.log(g_wh + 1e-5) / self.variances[1]
        self.res = torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]
        return self.res


class Decode:
    def __init__(self, loc: torch.Tensor, priors: torch.Tensor, variances: List[float]):
        self.loc = loc
        self.priors = priors
        self.variances = variances
        self.res: Optional[torch.Tensor] = None

    def __call__(self):
        self.res = torch.cat((
            self.priors[:, :2] + self.loc[:, :2] * self.variances[0] * self.priors[:, 2:],
            self.priors[:, 2:] * torch.exp(self.loc[:, 2:] * self.variances[1])), 1)
        self.res[:, :2] -= self.res[:, 2:] / 2
        self.res[:, 2:] += self.res[:, :2]
        return self.res


class LogSumExp:
    def __init__(self, x):
        self.x = x
        self.res: Optional[torch.Tensor] = None

    def __call__(self) -> torch.Tensor:
        x_max = self.x.data.max()
        self.res = torch.log(torch.sum(torch.exp(self.x - x_max), 1, keepdim=True)) + x_max
        return self.res


class NMS:
    def __init__(self, boxes: torch.Tensor, scores, overlap=0.5, top_k: int = 200):
        self.boxes: torch.Tensor = boxes
        self.scores: torch.Tensor = scores
        self.overlap = overlap
        self.top_k: int = top_k

        self.keep: Optional[torch.Tensor] = None
        self.count = None

    def __call__(self):
        self.keep = self.scores.new(self.scores.size()[0]).zero_().long()
        if self.boxes.numel() == 0:
            return self.keep
        x1, y1, x2, y2 = self.boxes[:, 0], self.boxes[:, 1], self.boxes[:, 2], self.boxes[:, 3]
        area: torch.Tensor = torch.mul(x2 - x1, y2 - y1)
        v, idx = self.scores.sort(0)
        idx = idx[-self.top_k:]
        xx1, yy1, xx2, yy2 = self.boxes.new(), self.boxes.new(), self.boxes.new(), self.boxes.new()
        # w, h = self.boxes.new(), self.boxes.new()

        self.count = 0
        while idx.numel() > 0:
            i = idx[-1]
            self.keep[self.count] = i
            self.count += 1
            if idx.size(0) == 1:
                break
            idx = idx[:-1]
            torch.index_select(x1, 0, idx, out=xx1)
            torch.index_select(y1, 0, idx, out=yy1)
            torch.index_select(x2, 0, idx, out=xx2)
            torch.index_select(y2, 0, idx, out=yy2)
            xx1, yy1 = torch.clamp(xx1, min=x1[i]), torch.clamp(yy1, min=y1[i])
            xx2, yy2 = torch.clamp(xx2, max=x2[i]), torch.clamp(yy2, max=y2[i])
            # w.resize_as_(xx2)
            # h.resize_as_(yy2)
            # w = xx2 - xx1
            # h = yy2 - yy1
            w = torch.clamp(xx2 - xx1, min=0.0)
            h = torch.clamp(yy2 - yy1, min=0.0)
            inter = w * h
            rem_areas = torch.index_select(area, 0, idx)
            union = (rem_areas - inter) + area[i]
            iou = inter / union
            idx = idx[iou.le(self.overlap)]
        return self.keep, self.count
