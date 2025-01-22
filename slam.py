import cv2
from display import Display

W = int(1920//2)
H = int(1080//2)

dis = Display(W, H)

def process_frame(img):
    img = cv2.resize(img, (dis.W, dis.H)) 
    dis.paint(img)

if __name__ == "__main__":
    
    cap = cv2.VideoCapture("test.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
        else:
            break