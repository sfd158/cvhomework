import torch
import torch.nn as nn
from torch.nn import functional
import logging
from typing import List, Tuple, Optional
import os
from utils.config import coco_refinedet, voc_refinedet
from utils.prior_box import PriorBox
from utils.l2norm import L2Norm
from utils.detection_refinedet import DetectRefineDet


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class RefineDet(nn.Module):
    def __init__(self, phase: str, size: int, base: List[nn.Module], extras: List[nn.Module],
                 arm: Tuple[List[nn.Conv2d], List[nn.Conv2d]],
                 odm: Tuple[List[nn.Conv2d], List[nn.Conv2d]],
                 tcb: Tuple[List[nn.Module], List[nn.Module], List[nn.Module]],
                 num_classes: int):
        super(RefineDet, self).__init__()
        self.phase: str = phase
        self.num_classes = num_classes
        self.cfg = (coco_refinedet, voc_refinedet)[num_classes == 21]
        self.priorbox: PriorBox = PriorBox(self.cfg[str(size)])
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.size = size

        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.conv4_3_L2Norm: L2Norm = L2Norm(512, 10)
        self.conv5_3_L2Norm: L2Norm = L2Norm(512, 8)
        self.extras = nn.ModuleList(extras)

        self.arm_loc = nn.ModuleList(arm[0])
        self.arm_conf = nn.ModuleList(arm[1])
        self.odm_loc = nn.ModuleList(odm[0])
        self.odm_conf = nn.ModuleList(odm[1])

        self.tcb0: nn.ModuleList = nn.ModuleList(tcb[0])
        self.tcb1: nn.ModuleList = nn.ModuleList(tcb[1])
        self.tcb2: nn.ModuleList = nn.ModuleList(tcb[2])
            
            
        self.sources_modnet0 = CALayer(512)
        self.sources_modnet1 = CALayer(512)
        self.sources_modnet2 = CALayer(1024)
        self.sources_modnet3 = CALayer(512)
        
        self.tcb_modnet0 = CALayer(256)
        self.tcb_modnet1 = CALayer(256)
        self.tcb_modnet2 = CALayer(256)
        self.tcb_modnet3 = CALayer(256)
        
        self.detect: Optional[DetectRefineDet] = None
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = DetectRefineDet(num_classes, self.size, 0, 1000, 0.01, 0.45, 0.01, 500)

    def vgg_forward(self, x: torch.Tensor):
        sources: List[torch.Tensor] = []
        # apply vgg up to conv4_3 relu and conv5_3 relu
        for k in range(30):
            x = self.vgg[k](x)
            if 22 == k:
                s = self.conv4_3_L2Norm(x)
                sources.append(s)
            elif 29 == k:
                s = self.conv5_3_L2Norm(x)
                sources.append(s)

        # apply vgg up to fc7
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = functional.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        return sources

    def tcb_forward(self, sources: List[torch.Tensor]):
        tcb_source: List[torch.Tensor] = []
        p: Optional[torch.Tensor] = None
        for k, v in enumerate(sources[::-1]):
            s: torch.Tensor = v
            for i in range(3):
                s = self.tcb0[(3 - k) * 3 + i](s)
            if k != 0:
                u: Optional[torch.Tensor] = p
                u = self.tcb1[3 - k](u)
                s += u
            for i in range(3):
                s = self.tcb2[(3 - k) * 3 + i](s)
            p = s
            tcb_source.append(s)
        tcb_source.reverse()
        return tcb_source

    def arm_loc_and_conf(self, sources: List[torch.Tensor]):
        arm_loc = []
        arm_conf = []
        for x, l, c in zip(sources, self.arm_loc, self.arm_conf):
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc_res = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf_res = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
        return arm_loc_res, arm_conf_res

    def odm_loc_and_conf(self, tcb_source):
        odm_loc = []
        odm_conf = []
        for x, l, c in zip(tcb_source, self.odm_loc, self.odm_conf):
            odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc_res = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        odm_conf_res = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)
        return odm_loc_res, odm_conf_res

    def forward(self, x: torch.Tensor):
        sources: List[torch.Tensor] = self.vgg_forward(x)
        
        sources[0] = self.sources_modnet0(sources[0])
        sources[1] = self.sources_modnet1(sources[1])
        sources[2] = self.sources_modnet2(sources[2])
        sources[3] = self.sources_modnet3(sources[3])
        
        arm_loc, arm_conf = self.arm_loc_and_conf(sources)
        
        tcb_source: List[torch.Tensor] = self.tcb_forward(sources)
        
        tcb_source[0] = self.tcb_modnet0(tcb_source[0])
        tcb_source[1] = self.tcb_modnet1(tcb_source[1])
        tcb_source[2] = self.tcb_modnet2(tcb_source[2])
        tcb_source[3] = self.tcb_modnet3(tcb_source[3])
        
        odm_loc, odm_conf = self.odm_loc_and_conf(tcb_source)
        
        if self.phase == "test":
            output = self.detect(
                arm_loc.view(arm_loc.size()[0], -1, 4),           # arm loc preds
                self.softmax(arm_conf.view(arm_conf.size()[0], -1, 2)),  # arm conf preds
                odm_loc.view(odm_loc.size()[0], -1, 4),           # odm loc preds
                self.softmax(odm_conf.view(odm_conf.size()[0], -1, self.num_classes)),   # odm conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                arm_loc.view(arm_loc.size()[0], -1, 4),
                arm_conf.view(arm_conf.size()[0], -1, 2),
                odm_loc.view(odm_loc.size()[0], -1, 4),
                odm_conf.view(odm_conf.size()[0], -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            logging.info('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            logging.info('Finished!')
        else:
            logging.warning('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
class VGGBuilder:
    base = {
        320: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
        512: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    }

    def __init__(self, size: int, i: int, batch_norm: bool = False):
        self.layers: List[nn.Module] = []
        self.cfg = self.base[size]
        self.i = i
        self.batch_norm = batch_norm

    def build(self) -> List[nn.Module]:
        in_channels = self.i
        for v in self.cfg:
            if v == 'M':
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif v == 'C':
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    self.layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    self.layers.extend([conv2d, nn.ReLU(inplace=True)])
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.layers.extend([pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)])
        return self.layers


class ExtrasBuilder:
    extras = {
        320: [256, 'S', 512],
        512: [256, 'S', 512],
    }

    def __init__(self, size: int, i, batch_norm: bool = False):
        self.cfg = self.extras[size]
        self.i = i
        self.batch_norm = batch_norm
        self.layers: List[nn.Conv2d] = []

    def build(self) -> List[nn.Conv2d]:
        # Extra layers added to VGG for feature scaling
        in_channels = self.i
        flag = False
        for k, v in enumerate(self.cfg):
            if in_channels != 'S':
                if v == 'S':
                    self.layers.append(nn.Conv2d(in_channels, self.cfg[k + 1],
                                                 kernel_size=(1, 3)[flag], stride=2, padding=1))
                else:
                    self.layers.append(nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag]))
                flag = not flag
            in_channels = v
        return self.layers


class ArmMultiBoxBuilder:

    mbox = {
        320: [3, 3, 3, 3],  # number of boxes per feature map location
        512: [3, 3, 3, 3],  # number of boxes per feature map location
    }

    def __init__(self, vgg: List[nn.Module], extra_layers, size: int):
        self.vgg = vgg
        self.extra_layers = extra_layers
        self.cfg = self.mbox[size]

    def build(self) -> Tuple[List[nn.Conv2d], List[nn.Conv2d]]:
        arm_loc_layers: List[nn.Conv2d] = []
        arm_conf_layers: List[nn.Conv2d] = []
        vgg_source = [21, 28, -2]
        for k, v in enumerate(vgg_source):
            arm_loc_layers.append(nn.Conv2d(self.vgg[v].out_channels, self.cfg[k] * 4, kernel_size=3, padding=1))
            arm_conf_layers.append(nn.Conv2d(self.vgg[v].out_channels, self.cfg[k] * 2, kernel_size=3, padding=1))
        for k, v in enumerate(self.extra_layers[1::2], 3):
            arm_loc_layers.append(nn.Conv2d(v.out_channels, self.cfg[k] * 4, kernel_size=3, padding=1))
            arm_conf_layers.append(nn.Conv2d(v.out_channels, self.cfg[k] * 2, kernel_size=3, padding=1))
        return arm_loc_layers, arm_conf_layers


class ODMMultiBoxBuilder:
    mbox = {
        320: [3, 3, 3, 3],  # number of boxes per feature map location
        512: [3, 3, 3, 3],  # number of boxes per feature map location
    }

    def __init__(self, vgg: List[nn.Module], extra_layers: List[nn.Module], size: int, num_classes: int):
        self.vgg = vgg
        self.cfg = self.mbox[size]
        self.extra_layers = extra_layers
        self.num_classes = num_classes
        self.vgg_source = [21, 28, -2]
        self.odm_loc_layers: List[nn.Conv2d] = []
        self.odm_conf_layers: List[nn.Conv2d] = []

    def build(self) -> Tuple[List[nn.Conv2d], List[nn.Conv2d]]:
        for k, v in enumerate(self.vgg_source):
            self.odm_loc_layers.append(nn.Conv2d(256, self.cfg[k] * 4, kernel_size=3, padding=1))
            self.odm_conf_layers.append(nn.Conv2d(256, self.cfg[k] * self.num_classes, kernel_size=3, padding=1))
        for k, v in enumerate(self.extra_layers[1::2], 3):
            self.odm_loc_layers.append(nn.Conv2d(256, self.cfg[k] * 4, kernel_size=3, padding=1))
            self.odm_conf_layers.append(nn.Conv2d(256, self.cfg[k] * self.num_classes, kernel_size=3, padding=1))
        return self.odm_loc_layers, self.odm_conf_layers


class TCBConstructor:
    tcb = {
        320: [512, 512, 1024, 512],
        512: [512, 512, 1024, 512],
    }

    def __init__(self, size: int):
        self.scale: List[nn.Module] = []
        self.upsample: List[nn.Module] = []
        self.pred_feature: List[nn.Module] = []
        self.cfg = self.tcb[size]

    def build(self) -> Tuple[List[nn.Module], List[nn.Module], List[nn.Module]]:
        for k, v in enumerate(self.cfg):
            self.scale.extend([nn.Conv2d(self.cfg[k], 256, 3, padding=1), nn.ReLU(inplace=True),
                               nn.Conv2d(256, 256, 3, padding=1)])
            self.pred_feature.extend([nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)])
            if k != len(self.cfg) - 1:
                self.upsample.append(nn.ConvTranspose2d(256, 256, 2, 2))
        return self.scale, self.upsample, self.pred_feature


def build_refinedet_modulation(phase: str, size: int = 320, num_classes=21):
    if phase != "test" and phase != "train":
        logging.error("Phase: " + phase + " not recognized")
        return
    if size != 320 and size != 512:
        logging.error(f"You specified size {size}. However, currently only RefineDet320 and RefineDet512 is supported!")
        return
    base: List[nn.Module] = VGGBuilder(size, 3).build()
    extras: List[nn.Conv2d] = ExtrasBuilder(size, 1024).build()
    arm = ArmMultiBoxBuilder(base, extras, size).build()
    odm = ODMMultiBoxBuilder(base, extras, size, num_classes).build()
    tcb = TCBConstructor(size).build()
    return RefineDet(phase, size, base, extras, arm, odm, tcb, num_classes)
