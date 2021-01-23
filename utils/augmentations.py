import torch
import cv2
import numpy as np
import types
from typing import List, Tuple, Optional


def intersect(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    max_xy: np.ndarray = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy: np.ndarray = np.maximum(box_a[:, :2], box_b[:2])
    inter: np.ndarray = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    inter = intersect(box_a, box_b)
    area_a: np.ndarray = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b: np.ndarray = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))  # [A,B]
    union: np.ndarray = area_a + area_b - inter
    return inter / union  # [A,B]


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels


class BaseAugType:
    def __init__(self):
        pass

    def __call__(self, img: np.ndarray, boxes: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None):
        return self.calc(img, boxes, labels)

    def calc(self, img: np.ndarray, boxes: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None):
        raise NotImplementedError


class Compose(BaseAugType):
    def __init__(self, trans: List[BaseAugType]):
        super(Compose, self).__init__()
        self.trans = trans

    def calc(self, img: np.ndarray, boxes=None, labels=None):
        for t in self.trans:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class LambdaAug(BaseAugType):
    def __init__(self, lambd: types.LambdaType):
        super(LambdaAug, self).__init__()
        self.lambd = lambd

    def calc(self, img: np.ndarray, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class FromIntsAug(BaseAugType):
    def __init__(self):
        super(FromIntsAug, self).__init__()

    def calc(self, img: np.ndarray, boxes=None, labels=None):
        return img.astype(np.float32), boxes, labels


class SubtractMeansAug(BaseAugType):
    def __init__(self, mean):
        super(SubtractMeansAug, self).__init__()
        self.mean: np.ndarray = np.array(mean, dtype=np.float32)

    def calc(self, img: np.ndarray, boxes=None, labels=None) \
            -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        img = img.astype(np.float32)
        img -= self.mean
        return img.astype(np.float32), boxes, labels


class ToAbsoluteCoordsAug(BaseAugType):
    def __init__(self):
        super(ToAbsoluteCoordsAug, self).__init__()

    def calc(self, img, boxes=None, labels=None):
        height, width, channels = img.shape
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        return img, boxes, labels


class ToPercentCoordsAug(BaseAugType):
    def calc(self, img: np.ndarray, boxes=None, labels=None):
        h, w, channels = img.shape
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        return img, boxes, labels


class ResizeAug(BaseAugType):
    def __init__(self, size: int = 300):
        super(ResizeAug, self).__init__()
        self.size: int = size

    def calc(self, img: np.ndarray, boxes=None, labels=None):
        img: np.ndarray = cv2.resize(img, (self.size, self.size))
        return img, boxes, labels


class RandAugSaturation(BaseAugType):
    def __init__(self, lower: float = 0.5, upper=1.5):
        super(RandAugSaturation, self).__init__()
        self.lower: float = lower
        self.upper: float = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def calc(self, img, boxes=None, labels=None):
        if np.random.randint(2):
            img[:, :, 1] *= np.random.uniform(self.lower, self.upper)
        return img, boxes, labels


class RandomAugHue(BaseAugType):
    def __init__(self, delta=18.0):
        super(RandomAugHue, self).__init__()
        assert 0.0 <= delta <= 360.0
        self.delta: float = delta

    def calc(self, img, boxes=None, labels=None):
        if np.random.randint(2):
            img[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return img, boxes, labels


class RandAugLightNoise(BaseAugType):
    def __init__(self):
        super(RandAugLightNoise, self).__init__()
        self.perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))

    def calc(self, img: np.ndarray, boxes=None, labels=None) \
            -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        if np.random.randint(2):
            swap_res: Tuple[int, int, int] = self.perms[np.random.randint(len(self.perms))]
            shuffle_res = SwapChannels(swap_res)
            img = shuffle_res(img)
        return img, boxes, labels


class AugConvertColor(BaseAugType):
    def __init__(self, current='BGR', transform='HSV'):
        super(AugConvertColor, self).__init__()
        self.trans: str = transform
        self.curr: str = current

    def calc(self, img: np.ndarray, boxes=None, labels=None):
        if self.curr == 'BGR' and self.trans == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.curr == 'HSV' and self.trans == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return img, boxes, labels


class RandomContrast(BaseAugType):
    def __init__(self, lower=0.5, upper=1.5):
        super(RandomContrast, self).__init__()
        self.lower: float = lower
        self.upper: float = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def calc(self, img: np.ndarray, boxes=None, labels=None):
        if np.random.randint(2):
            alpha: float = np.random.uniform(self.lower, self.upper)
            img *= alpha
        return img, boxes, labels


class AugRandomBrightness(BaseAugType):
    def __init__(self, delta=32.0):
        super(AugRandomBrightness, self).__init__()
        assert 0.0 <= delta <= 255.0
        self.delta: float = delta

    def calc(self, img: np.ndarray, boxes: Optional[np.ndarray] = None, labels=None):
        if np.random.randint(2):
            delta: float = np.random.uniform(-self.delta, self.delta)
            img += delta
        return img, boxes, labels


class ToCV2img:
    def __call__(self, tensor: torch.Tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor:
    def __call__(self, cvimg: np.ndarray, boxes=None, labels=None):
        return torch.from_numpy(cvimg.astype(np.float32)).permute(2, 0, 1), boxes, labels


class AugRandomSampleCrop(BaseAugType):
    def __init__(self):
        super(AugRandomSampleCrop, self).__init__()
        self.sample_options = (
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        )

    def calc(self, img: np.ndarray, boxes=None, labels=None):
        height, width, _ = img.shape
        while True:
            mode = np.random.choice(self.sample_options)
            if mode is None:
                return img, boxes, labels

            min_iou, max_iou = mode
            min_iou = float('-inf') if min_iou is None else min_iou
            max_iou = float('inf') if max_iou is None else max_iou

            for _ in range(50):
                curr_img: np.ndarray = img
                w: float = np.random.uniform(0.3 * width, width)
                h: float = np.random.uniform(0.3 * height, height)
                if h / w < 0.5 or h / w > 2:
                    continue

                left: float = np.random.uniform(width - w)
                top: float = np.random.uniform(height - h)
                rect: np.ndarray = np.array([int(left), int(top), int(left + w), int(top + h)])
                overlap: np.ndarray = jaccard_numpy(boxes, rect)
                if np.min(overlap) < min_iou and max_iou < np.max(overlap):
                    continue
                curr_img: np.ndarray = curr_img[rect[1]:rect[3], rect[0]:rect[2], :]
                center_pos: np.ndarray = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                m1: np.ndarray = np.asarray(rect[0] < center_pos[:, 0]) * (rect[1] < center_pos[:, 1])
                m2: np.ndarray = np.asarray(rect[2] > center_pos[:, 0]) * (rect[3] > center_pos[:, 1])

                mask: np.ndarray = m1 * m2
                if not mask.any():
                    continue

                curr_boxes: np.ndarray = boxes[mask, :].copy()
                curr_labels = labels[mask]
                curr_boxes[:, :2] = np.maximum(curr_boxes[:, :2], rect[:2]) - rect[:2]
                curr_boxes[:, 2:] = np.minimum(curr_boxes[:, 2:], rect[2:]) - rect[:2]

                return curr_img, curr_boxes, curr_labels


class AugExpand(BaseAugType):
    def __init__(self, mean):
        super(AugExpand, self).__init__()
        self.mean = mean

    def calc(self, img: np.ndarray, boxes=None, labels=None):
        if np.random.randint(2):
            return img, boxes, labels

        h, w, dep = img.shape
        ratio = np.random.uniform(1, 4)
        left = np.random.uniform(0, w * ratio - w)
        top = np.random.uniform(0, h * ratio - h)

        expand_img = np.zeros((int(h * ratio), int(w * ratio), dep), dtype=img.dtype)
        expand_img[:, :, :] = self.mean
        expand_img[int(top):int(top + h), int(left):int(left + w)] = img
        img = expand_img

        boxes = boxes.copy()
        boxes[:, :2] += np.array([int(left), int(top)])
        boxes[:, 2:] += np.array([int(left), int(top)])

        return img, boxes, labels


class AugRandomMirror(BaseAugType):
    def __init__(self):
        super(AugRandomMirror, self).__init__()

    def calc(self, img: np.ndarray, boxes=None, labels=None):
        _, width, _ = img.shape
        if np.random.randint(2):
            img = img[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return img, boxes, labels


class SwapChannels:
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, img: np.ndarray):
        img: np.ndarray = img[:, :, self.swaps]
        return img


class AugPhotometricDistort(BaseAugType):
    def __init__(self):
        super(AugPhotometricDistort, self).__init__()
        self.pd = [
            RandomContrast(),
            AugConvertColor(transform='HSV'),
            RandAugSaturation(),
            RandomAugHue(),
            AugConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = AugRandomBrightness()
        self.rand_light_noise = RandAugLightNoise()

    def calc(self, img, boxes=None, labels=None):
        im = img.copy()
        im, boxes, labels = self.rand_brightness.calc(im, boxes, labels)
        if np.random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort.calc(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class SSDAugment(BaseAugType):
    def __init__(self, size: int = 300, mean=(104, 117, 123)):
        super(SSDAugment, self).__init__()
        self.mean = mean
        self.size = size
        self.augment = Compose([
            FromIntsAug(),
            ToAbsoluteCoordsAug(),
            AugPhotometricDistort(),
            AugExpand(self.mean),
            AugRandomSampleCrop(),
            AugRandomMirror(),
            ToPercentCoordsAug(),
            ResizeAug(self.size),
            SubtractMeansAug(self.mean)
        ])

    def __call__(self, img, boxes=None, labels=None):
        return self.calc(img, boxes, labels)

    def calc(self, img, boxes=None, labels=None):
        res = self.augment.calc(img, boxes, labels)
        return res
