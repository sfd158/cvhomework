import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os
import time
import argparse
import numpy as np
import xml.etree.ElementTree as ET

from utils.augmentations import BaseTransform
from utils.voc0712 import VOC_ROOT, VOCDetection, VOCAnnotationTransform
from utils.voc0712 import VOC_CLASSES as labelmap
from utils.config import MEANS


def str2bool(v: str):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default="refinedet", choices=["refineDet", "deform", "modulation"], type=str)
parser.add_argument('--trained_model', default='./weights/RefineDet320_VOC_final.pth', type=str)
parser.add_argument('--save_folder', default='eval/', type=str)
parser.add_argument('--confidence_threshold', default=0.01, type=float)
parser.add_argument('--top_k', default=5, type=int)
parser.add_argument('--cuda', default=True, type=str2bool)
parser.add_argument('--voc_root', default=VOC_ROOT)
parser.add_argument('--cleanup', default=True, type=str2bool)
parser.add_argument('--input_size', default='320', choices=['320', '512'], type=str)

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets', 'Main', '%s.txt')
YEAR = '2007'
devkit_path = args.voc_root + 'VOC' + YEAR
set_type = 'test'


class Timer:
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class RecParser:
    def __call__(self, filename: str):
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {'name': obj.find('name').text, 'pose': obj.find('pose').text,
                          'truncated': int(obj.find('truncated').text), 'difficult': int(obj.find('difficult').text)}
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1, int(bbox.find('ymin').text) - 1,
                                  int(bbox.find('xmax').text) - 1, int(bbox.find('ymax').text) - 1]
            objects.append(obj_struct)

        return objects


def get_output_dir(name, phase):
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


class ResFileFormat:
    def __call__(self, image_set, cls):
        filename = 'det_' + image_set + '_%s.txt' % (cls)
        filedir = os.path.join(devkit_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path


class ResWriter:
    def __call__(self, all_boxes, dataset):
        for cls_ind, cls in enumerate(labelmap):
            print('Writing {:s} VOC results file'.format(cls))
            filename = ResFileFormat()(set_type, cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(dataset.ids):
                    dets = all_boxes[cls_ind+1][im_ind]
                    if len(dets) == 0:
                        continue

                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1], dets[k, 0] + 1,
                                       dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))


class EvalClass:
    def __call__(self, output_dir='output', use_07=True):
        cachedir = os.path.join(devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(labelmap):
            filename = ResFileFormat()(set_type, cls)
            rec, prec, ap = VOCEval()(
               filename, annopath, imgsetpath % set_type, cls, cachedir, 0.5, use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.txt'), 'w') as f:
                f.write(str({'rec': np.asarray(rec).tolist(), 'prec': np.asarray(prec).tolist(),
                             'ap': np.asarray(ap).tolist()}))

        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))


class VOCAP:
    def __call__(self, rec, prec, use_07_metric=True):
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


class VOCEval:
    @staticmethod
    def get_image_names(imgsetfile: str):
        with open(imgsetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        return imagenames

    def get_class_recs(self, imagenames, recs, classname):
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox, 'difficult': difficult, 'det': det}
        return npos, class_recs

    def __call__(self, detpath, annopath, imagesetfile, classname, cachedir, ovthresh=0.5, use_07_metric=True):
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        imagenames = self.get_image_names(imagesetfile)
        recs = {imagename: RecParser()(annopath % imagename) for imagename in imagenames}
        npos, class_recs = self.get_class_recs(imagenames, recs, classname)

        # read dets
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:

            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            # sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp, fp = np.zeros(nd), np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                           (BBGT[:, 2] - BBGT[:, 0]) *
                           (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = VOCAP()(rec, prec, use_07_metric)
        else:
            rec, prec, ap = -1.0, -1.0, -1.0

        return rec, prec, ap


class NetTest:
    def __call__(self, net, dataset):
        num_images = len(dataset)
        all_boxes = [[np.array([]) for _ in range(num_images)] for _ in range(len(labelmap)+1)]

        _t = {'im_detect': Timer(), 'misc': Timer()}
        output_dir = get_output_dir('refinedet-eval', set_type)
        det_file = os.path.join(output_dir, 'detections.txt')

        for i in range(num_images):
            im, gt, h, w = dataset.pull_item(i)

            x = Variable(im.unsqueeze(0))
            if args.cuda:
                x = x.cuda()
            _t['im_detect'].tic()
            detections = net(x).data
            detect_time = _t['im_detect'].toc(average=False)

            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32, copy=False)
                all_boxes[j][i] = cls_dets

            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

        with open(det_file, 'w') as f:
            f.write(str(all_boxes))

        print('Evaluating detections')
        self.evaluate_detections(all_boxes, output_dir, dataset)

    @staticmethod
    def evaluate_detections(box_list, output_dir, dataset):
        ResWriter()(box_list, dataset)
        EvalClass()(output_dir)


def main():
    num_classes = len(labelmap) + 1

    if args.model_type == "refinedet":
        from models.refinedet import build_refinedet
        net = build_refinedet('test', int(args.input_size), num_classes)
    elif args.model_type == "deform":
        from models.refinedet_deform import build_refinedet_deform
        net = build_refinedet_deform('test', int(args.input_size), num_classes)
    elif args.model_type == "modulation":
        from models.refinedet_modulation import build_refinedet_modulation
        net = build_refinedet_modulation('test', int(args.input_size), num_classes)
    else:
        raise NotImplementedError

    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load datasets
    dataset = VOCDetection(args.voc_root, [('2007', set_type)], BaseTransform(int(args.input_size), MEANS),
                           VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    with torch.no_grad():
        NetTest()(net, dataset)


if __name__ == '__main__':
    main()
