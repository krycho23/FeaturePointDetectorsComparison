import cv2
from matplotlib import pyplot as plt
import numpy as np
import json
import math
import os
import sys
import re


class CornerBasedMatching:

    def __init__(self,  blockSize, ksize):
        self.scale = 1
        left_img = "Aloe/view1.png"
        right_img = "Aloe/view5.png"

        # Load source image
        self.img1 = cv2.imread(left_img, cv2.IMREAD_COLOR)  # Load an image

        # Check if image is loaded fine
        if self.img1 is None:
            print('Error opening image1')
            exit(-1)

        # Reduce noise
        # Remove noise by blurring with a Gaussian filter
        self.img1 = cv2.GaussianBlur(self.img1, (3, 3), 0)

        # Convert the image to grayscale
        self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)

        self.img2 = cv2.imread(right_img, cv2.IMREAD_COLOR)  # Load an image

        # Check if image is loaded fine
        if self.img2 is None:
            print('Error opening image2')
            exit(-1)

        # Reduce noise
        # Remove noise by blurring with a Gaussian filter
        self.img2 = cv2.GaussianBlur(self.img2, (3, 3), 0)

        # Convert the image to grayscale
        self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)


        self.total_cols = len(self.img1)
        self.total_rows = len(self.img1[0])
        self.ground_truth = np.zeros((self.total_cols, self.total_rows), np.uint8)
        self.disparity = np.zeros((self.total_cols, self.total_rows), np.uint8)
        self.error_map = np.zeros((self.total_cols, self.total_rows), np.uint8)

    # calculate factor calculated where realative_error is smaller than delta = delta[%]/100= 0.20 divided by all points
    def calculateMatchFactor(self, relative_list, delta):

        counter = 0
        for relative in relative_list:
            if relative > delta and math.isnan(relative) == False:
                counter += 1

        # calculate factor
        return counter / len(relative_list)

    # compare calculated values with ground-truth values
    def compareGroundtruth(self):

        # Load ground-truth
        ground_name = "Aloe/3dGT.png"
        self.ground_truth =  cv2.imread(ground_name, cv2.IMREAD_GRAYSCALE)

        # normalization of disp values for rescaled third-size images
        self.disparity = self.disparity / self.scale

        left_corners = self.calculateCorners()

        error_list = []

        # comapare diff between disp - disp_ground
        for left_cross in left_corners:
            x = left_cross[0]
            y = left_cross[1]

            error = abs(self.ground_truth[y, x] - self.disparity[y, x])
            error_list.append(error)
            self.error_map[y, x] = error

        return self.calculateMatchFactor(error_list, 5 / self.scale)

    # def calculateCorners(self, src, blockSize, ksize, k):
    def calculateCorners(self):

        blockSize = 7
        ksize = 3
        k = 0.04

        dst = cv2.cornerHarris(self.img1, blockSize, ksize, k)
        ret, final = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)

        corner_value = 0
        corners = []

        # high is y, width is x
        for i in range(len(final)):
            for j in range(len(final[0])):
                if corner_value == final[i, j]:
                    corners.append([j, i])

        return corners

    def doMatching(self):
        left_corners = self.calculateCorners()

        minDisparity = 0
        numDisparities = 128
        blockSize = 11
        uniquenessRatio = 1
        speckleWindowSize = 3
        speckleRange = 3
        disp12MaxDiff = 200
        P1 = 600
        P2 = 2400
        speckleRange = 3
        disp12MaxDiff = 200

        stereo = cv2.StereoSGBM_create(
        minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        uniquenessRatio = uniquenessRatio,
        speckleRange = speckleRange,
        speckleWindowSize = speckleWindowSize,
        disp12MaxDiff = disp12MaxDiff,
        P1 = P1,
        P2 = P2)
        self.disparity = stereo.compute(self.img1, self.img2)
      
        match_factor = self.compareGroundtruth()
        
        plt.figure(figsize=(20, 10))
        plt.bar("MATCH FACTOR", match_factor)
        plt.savefig("3dOutput/shi_tomasi_Aloe3d_" + str(blockSize) + "_" + str(ksize) + "_" + str(1) + ".png")
        
        
        
        


if __name__ == '__main__':

    if len(sys.argv) != 3:
            print("Wrong number of arguments")
            exit(1)
            
    blockSize = int(sys.argv[1])
    ksize = int(sys.argv[2])

    matching = CornerBasedMatching(blockSize, ksize)
    matching.doMatching()

    
