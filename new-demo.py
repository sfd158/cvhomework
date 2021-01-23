import os
import sys
from matplotlib import pyplot as plt
import torch
import numpy as np
import cv2

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


from models.refinedet import build_refinedet
from utils.voc0712 import VOC_ROOT, VOCDetection, VOCAnnotationTransform
from utils.voc0712 import VOC_CLASSES as labels


def main():
    net = build_refinedet('test', 320, 21)
    net.load_weights('./weights/RefineDet320_VOC_final.pth')

    # year (07 or 12) and dataset ('test', 'val', 'train')
    testset = VOCDetection(VOC_ROOT, [('2007', 'val')], None, VOCAnnotationTransform(), use_buf=False)

    img_id = 102
    image = testset.pull_image(img_id)
    rgb_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    # plt.show()

    x: np.ndarray = cv2.resize(image, (320, 320)).astype(np.float32)
    x -= np.array([104.0, 117.0, 123.0], dtype=np.float32)
    x = x[:, :, ::-1].copy()

    xx = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    xx = xx.cuda() if torch.cuda.is_available() else xx
    with torch.no_grad():
        y = net(xx)

    plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21))
    plt.imshow(rgb_image)
    current_axis = plt.gca()

    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            label_name = labels[i - 1]
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            color = colors[i]
            current_axis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            current_axis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
            j += 1
    plt.savefig(f"{img_id}result.png")
    plt.show()


if __name__ == "__main__":
    main()
