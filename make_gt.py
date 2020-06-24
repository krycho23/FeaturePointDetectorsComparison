import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('Aloe/view1.png',0)

# Check if image is loaded fine
if img is None:
    print('Error opening image1')
    exit(-1)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create(threshold=25)

# find and draw the keypoints
kp = fast.detect(img,None)
print(kp[0].pt[0])
x = int(kp[0].pt[0])
y = int(kp[0].pt[1])

img2 = cv2.drawKeypoints(img, kp, None,color=(255,0,0))

print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Total Keypoints with nonmaxSuppression: ", len(kp))

cv2.imwrite('Aloe/2dGT.png',img2)

