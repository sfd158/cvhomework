import sys
import os
import torch
from torch.autograd import Variable
import cv2
import argparse
from models.refinedet import build_refinedet
from utils.augmentations import BaseTransform
from utils.voc0712 import VOC_CLASSES as labelmap


COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

use_cuda = True
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["video", "camera"], default="video")
parser.add_argument("--render", type=str, choices=["imshow", "save"], default="save")


def predict(frame):
    height, width = frame.shape[:2]
    x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if use_cuda:
        x = x.cuda()
    y = net(x)  # forward pass
    detections = y.detach().cpu()
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height]).cpu()
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), COLORS[i % 3], 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                        FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame


if __name__ == "__main__":
    args = parser.parse_args()
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    net = build_refinedet('test', 320, 21).cuda()
    net.load_weights('./weights/RefineDet320_VOC_final.pth')
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

    if args.mode == "camera":
        camera_number = 0
        cap = cv2.VideoCapture(camera_number + cv2.CAP_DSHOW)  # use camera
    elif args.mode == "video":
        cap = cv2.VideoCapture("demo-input-short.mp4")
    else:
        raise NotImplementedError

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('testwrite.avi', fourcc, 20.0, (1920, 1080), True)
    out = cv2.VideoWriter("demo-output-short.avi", fourcc, 25, (1920, 1080), True) if args.render == "save" else None

    for i in range(1000):
        ret, frame = cap.read()
        print(f"Frame {i}")
        if not ret:
            break
        with torch.no_grad():
            frame = predict(frame)

        if args.render == "imshow":
            cv2.imshow("capture", frame)
        else:
            out.write(frame)
        if cv2.waitKey(100) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()
    # if out:
    #    out.release()
