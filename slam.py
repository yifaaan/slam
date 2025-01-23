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
    matches = fe.extract(img)

    print("%d matches" % len(matches))
    # for p in kps:
    #     x,y = map(lambda x: int(round(x)), p.pt)
    #     cv2.circle(img, (x, y), color=(0, 255, 0), radius=5)
    for query_pt, train_pt in matches:
        # print(query_pt)
        x1,y1 = map(lambda x: int(round(x)), query_pt)
        x2,y2 = map(lambda x: int(round(x)), train_pt)
        cv2.circle(img, (x1, y1), color=(0, 255, 0), radius=3)
        cv2.line(img, (x1, x2), (x2, y2), color=(255,0,0))
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