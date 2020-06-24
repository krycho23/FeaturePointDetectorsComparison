import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import sys


class CornerBasedMatching:

    def __init__(self, _quality_level, _euclides_distance):
        self.scale = 1
        self.quality_level = _quality_level
        self.euclides_distance = _euclides_distance
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
    def calculateMatchFactor(self, error_list, delta):

        counter = 0
        for error in error_list:
            if error > delta and math.isnan(error) == False:
                counter += 1

        # calculate factor
        return counter / len(error_list)

    # compare calculated values with ground-truth values
    def compareGroundtruth(self):

        # Load ground-truth
        ground_name = "Aloe/3dGT.png"
        self.ground_truth =  cv2.imread(ground_name, cv2.IMREAD_GRAYSCALE)

        # normalization of disp values for rescaled third-size images
        self.disparity = self.disparity / self.scale

        left_corners = self.calculateCorners()
        
        error_list = []
        for left_corner in left_corners:
            x, y = left_corner.ravel()
            error = abs(self.disparity[y, x] - self.ground_truth[y, x])
            error_list.append(error)
            
        return self.calculateMatchFactor(error_list, 10 / self.scale)

    # def calculateCorners(self, src, blockSize, ksize, k):
    def calculateCorners(self):

        corners = cv2.goodFeaturesToTrack(self.img1, 25, self.quality_level, self.euclides_distance)
        corners = np.int0(corners)

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
        plt.savefig("3dOutput/shi_tomasi_Aloe3d_" + str(self.quality_level) + "_" + str(self.euclides_distance) + "_" + str(1) + ".png")


if __name__ == '__main__':
        
        if len(sys.argv) != 3:
            print("Wrong number of arguments")
            exit(1)
            
        quality_level = float(sys.argv[1])
        euclides_distance = int(sys.argv[2])

        matching = CornerBasedMatching(quality_level, euclides_distance)
        matching.doMatching()
    
