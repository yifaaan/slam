import cv2
import numpy as np

class Extractor(object):
    # dividing the input image into GX x GY (16 x 12) smaller chunks
    GX = 16//2
    GY = 12//2

    def __init__(self):
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher()
        self.last = None

    def extract(self, img):
        # run detect in grid
        # print("img.shape:", img.shape) (540, 960, 3)
        # sy = img.shape[0]//self.GY
        # sx = img.shape[1]//self.GX
        # print("sy:", sy, "sx:", sx)
        # akp = []
        # for ry in range(0, img.shape[0], sy):
        #     for rx in range(0, img.shape[1], sx):
        #         img_chunk = img[ry:ry+sy, rx:rx+sx]
        #         # print(img_chunk.shape)
        #         kps = self.orb.detect(img_chunk, None)
        #         for p in kps:
        #             p.pt = (p.pt[0] + rx, p.pt[1] + ry)
        #             akp.append(p)
        # return akp


        # detection 
        kps =  cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, 0.01, 3.0)

        # extraction
        # kps =  self.orb.detect(img)
        kps = [cv2.KeyPoint(x=p[0][0], y=p[0][1], size=20) for p in kps]
        kps, des = self.orb.compute(img, kps)


        # matching
        matches = None
        if self.last is not None:
            m = self.bf.match(des, self.last['des'])
            matches = zip([kps[m.queryIdx] for m in matches], [self.last['kps'][m.trainIdx] for m in matches]) 

        self.last = {'kps': kps, 'des': des}

        return matches 
