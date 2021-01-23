from utils.config import HOME
import os
import tqdm
import logging
import torch
import torch.utils.data as data
import cv2
import numpy as np
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

VOC_ROOT = os.path.join(HOME, "./datasets/VOCdevkit/")


class VOCAnnotationTransform(object):
    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1

                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCBufType:
    def __init__(self):
        self.im = []
        self.gt = []
        self.h = []
        self.w = []

    def append(self, im, gt, h, w):
        self.im.append(im)
        self.gt.append(gt)
        self.h.append(h)
        self.w.append(w)

    def __getitem__(self, index):
        # im, gt, h, w = self.pull_item(index)
        return self.im[index].clone(), self.gt[index].copy()


class VOCDetection(data.Dataset):
    def __init__(self, root: str,
                 image_sets: Optional[List[Tuple[str, str]]] = None,
                 transform=None, target_trans=VOCAnnotationTransform(),
                 dataset_name='VOC0712', use_buf=True):
        if image_sets is None:
            image_sets = [('2007', 'trainval'), ('2012', 'trainval')]
        self.root: str = root
        self.image_set = image_sets
        self.transform = transform
        self.target_trans = target_trans
        self.name = dataset_name
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = []
        for year, name in image_sets:
            root_path = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(root_path, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((root_path, line.strip()))

        self.buf: Optional[VOCBufType] = None if not use_buf else VOCBufType()
        if use_buf:
            logging.info(f"Use Buffer in DataSet, len = {len(self)}.")
            for index in tqdm.tqdm(range(len(self.ids))):
                im, gt, h, w = self.pull_item(index)
                self.buf.append(im, gt, h, w)

    def __getitem__(self, index):
        if self.buf is None:
            im, gt, h, w = self.pull_item(index)
            return im, gt
        else:
            # logging.debug("use buf")
            return self.buf[index]

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_trans is not None:
            target = self.target_trans(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

            img = img[:, :, (2, 1, 0)]  # # to rgb
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index) -> np.ndarray:
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_trans(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        return torch.from_numpy(self.pull_image(index)).unsqueeze_(0)
