import cv2
import numpy as np
from display import Display
from extractor import Extractor


W = int(1920//2)
H = int(1080//2)

dis = Display(W, H)


fe = Extractor()

def process_frame(img):
    img = cv2.resize(img, (dis.W, dis.H))
    kps, des, matches = fe.extract(img)
    if matches is None:
        return

    # for p in kps:
    #     x,y = map(lambda x: int(round(x)), p.pt)
    #     cv2.circle(img, (x, y), color=(0, 255, 0), radius=5)
    for p in kps:
        x,y = map(lambda x: int(round(x)), p.pt)
        cv2.circle(img, (x, y), color=(0, 255, 0), radius=3)

    dis.paint(img)


if __name__ == "__main__":
    
    cap = cv2.VideoCapture("test.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        # frame.shape(height, width) (1080, 1920, 3)
        # print("init.img.shape:", frame.shape)
        if ret:
            process_frame(frame)
        else:
            break