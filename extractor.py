import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
class Extractor(object):
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None

    def extract(self, img):
        # detection 
        kps =  cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, 0.01, 3.0)

        # extraction
        kps = [cv2.KeyPoint(x=p[0][0], y=p[0][1], size=20) for p in kps]
        kps, des = self.orb.compute(img, kps)


        # matching
        ret = [] 
        if self.last is not None:
            ms = self.bf.knnMatch(des,self.last['des'], k=2)
            for m,n in ms:
                if m.distance < 0.75*n.distance:
                    kp_1 =  kps[m.queryIdx].pt
                    kp_2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp_1, kp_2))
            
        
        if len(ret) > 0:
            ret = np.array(ret)
            print(ret.shape)
            model, inliers = ransac((ret[:,0], ret[:,1]),
                                    FundamentalMatrixTransform,
                                    min_samples=8, 
                                    residual_threshold=0.01,
                                    max_trials=100) 
            ret = ret[inliers]
        self.last = {'kps': kps, 'des': des}

        # F, mask = cv2.findFundamentalMat(kps_q, kps_t, cv2.FM_8POINT)
        return ret 
