import torch
from typing import List
from utils.box_utils import CenterSize, Decode, NMS
from utils.config import voc_refinedet as cfg


class DetectRefineDet(object):
    def __init__(self, num_classes: int, size: int, bkg_label: int, top_k: int, conf_thresh: float, nms_thresh: float,
                 objectness_thre: float, keep_top_k: int):
        super(DetectRefineDet, self).__init__()
        self.num_classes: int = num_classes
        self.background_label: int = bkg_label
        self.top_k: int = top_k
        self.keep_top_k: int = keep_top_k
        # Parameters used in nms.
        self.nms_thresh: float = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh: float = conf_thresh
        self.objectness_thre: float = objectness_thre
        self.variance: List[float] = cfg[str(size)]['variance']

    def bboxes(self, num: int, arm_loc_data, prior_data, odm_loc_data, conf_preds):
        output: torch.Tensor = torch.zeros(num, self.num_classes, self.top_k, 5)
        for i in range(num):
            default: torch.Tensor = Decode(arm_loc_data[i], prior_data, self.variance)()
            default: torch.Tensor = CenterSize(default)()
            decoded_boxes: torch.Tensor = Decode(odm_loc_data[i], default, self.variance)()
            conf_scores: torch.Tensor = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask: torch.Tensor = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes: torch.Tensor = decoded_boxes[l_mask].view(-1, 4)
                ids, count = NMS(boxes, scores, self.nms_thresh, self.top_k)()
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
        return output

    def forward(self, *args) -> torch.Tensor:
        arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, prior_data = args
        arm_object_conf: torch.Tensor = arm_conf_data.data[:, :, 1:]
        no_object_index: torch.Tensor = torch.as_tensor(arm_object_conf <= self.objectness_thre)
        odm_conf_data[no_object_index.expand_as(odm_conf_data)] = 0

        num: int = odm_loc_data.size(0)  # batch size
        num_priors: int = prior_data.size(0)
        conf_preds: torch.Tensor = odm_conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        output: torch.Tensor = self.bboxes(num, arm_loc_data, prior_data, odm_loc_data, conf_preds)
        flt: torch.Tensor = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.keep_top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output

    def __call__(self, *args) -> torch.Tensor:
        return self.forward(*args)
