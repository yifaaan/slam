import cv2
import numpy as np
from display import Display
from extractor import Extractor


W = int(1920//2)
H = int(1080//2)

dis = Display(W, H)
F = 1
K = np.array(([F,0,W//2],[0,F,H//2],[0, 0, 1]))
print(K)
fe = Extractor(K)

def process_frame(img):
    img = cv2.resize(img, (dis.W, dis.H))
    matches = fe.extract(img)
    if len(matches) == 0:
        return
    print("%d matches" % len(matches))

    
    for query_pt, train_pt in matches:
        x1,y1 = fe.denormalize(query_pt) 
        x2,y2 = fe.denormalize(train_pt) 

        cv2.circle(img, (x1, y1), color=(0, 255, 0), radius=3)
        cv2.line(img, (x1, y1), (x2, y2), color=(255,0,0))
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