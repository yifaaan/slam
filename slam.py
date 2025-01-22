import cv2
from display import Display

W = int(1920//2)
H = int(1080//2)

dis = Display(W, H)
orb = cv2.ORB_create()

def process_frame(img):
    img = cv2.resize(img, (dis.W, dis.H)) 
    kp, des = orb.detectAndCompute(img,None)
    for p in kp:
        x,y = map(lambda x: int(round(x)), p.pt)
        cv2.circle(img, (x, y), color=(0, 255, 0), radius=3)
    dis.paint(img)


if __name__ == "__main__":
    
    cap = cv2.VideoCapture("test.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
        else:
            break