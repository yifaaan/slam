import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
class Extractor(object):
    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K 
        self.Kinv = np.linalg.inv(self.K)

    def denormalize(self, pt):
        return np.dot(self.Kinv, [pt[0], pt[1], 1.0])
        # return int(round(pt[0] + self.w // 2)), int(round(pt[1] + self.h // 2))

    def extract(self, img):
        # detection 
        kps =  cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3.0)

        # extraction
        kps = [cv2.KeyPoint(x=p[0][0], y=p[0][1], size=20) for p in kps]
        kps, des = self.orb.compute(img, kps)


        # matching
        ret = [] 
        if self.last is not None:
            ms = self.bf.knnMatch(des,self.last['des'], k=2)
            for m,n in ms:
                if m.distance < 0.75*n.distance:
                    kp1 =  kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))
            
        
        # filter
        if len(ret) > 0:
            ret = np.array(ret)
            print(ret.shape)

            # normalize coords: subtract to move to 0
            ret[:, :, 0] -= img.shape[0]//2
            ret[:, :, 1] -= img.shape[1]//2

            model, inliers = ransac((ret[:,0], ret[:,1]),
                                    FundamentalMatrixTransform,
                                    min_samples=8, 
                                    residual_threshold=1,
                                    max_trials=100) 
            ret = ret[inliers]
            s,v,d = np.linalg.svd(model.params)
            print(v)
        self.last = {'kps': kps, 'des': des}
       
        return ret