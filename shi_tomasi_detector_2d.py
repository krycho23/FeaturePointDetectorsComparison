import cv2
import json
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import sys

if len(sys.argv) != 3:
    print("Wrong number of arguments")
    exit(1)

left_img = "Aloe/view1.png"
# Load source image
img1 = cv2.imread(left_img, cv2.IMREAD_COLOR)  # Load an image

# Check if image is loaded fine
if img1 is None:
    print('Error opening image1')
    exit(-1)

# Reduce noise
# Remove noise by blurring with a Gaussian filter
img1 = cv2.GaussianBlur(img1, (3, 3), 0)

# Convert the image to grayscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

total_cols = len(img1)
total_rows = len(img1[0])

qualityLevel = float(sys.argv[1])
euclidesDistance = int(sys.argv[2])


# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create(threshold=25)

# find and draw the keypoints
kp = fast.detect(img1,None)
ground_truth = np.ones(shape=(img1.shape))

for point in kp:
	x = int(kp[0].pt[0])
	y = int(kp[0].pt[1])
	ground_truth[y,x] = 2

#visualise ground-truth
total_cols = len(ground_truth[0])
total_rows = len(ground_truth)

black = 255
white = 0

ground_truth_v = np.zeros((total_rows, total_cols), np.uint8)

corner_number = 0
for row in range(total_rows):
    for col in range(total_cols):
        if ground_truth[row,col] == 1:
            ground_truth_v[row, col] = white
        elif ground_truth[row,col] == 2:
            ground_truth_v[row,col] = black
            corner_number +=1


corners = cv2.goodFeaturesToTrack(img1, 25, qualityLevel, euclidesDistance)
corners = np.int0(corners)

final = np.zeros((total_rows, total_cols), np.uint8)

for i in corners:
    x, y = i.ravel()
    final[y,x] = 255

# false-positives
false_positives = 0
for row in range(total_rows):
    for col in range(total_cols):
        if ground_truth[row,col] == 1 and final[row,col] == 255:
            false_positives +=1


# false-negatives
false_negatives = 0
for row in range(total_rows):
    for col in range(total_cols):
        if ground_truth[row, col] == 2 and final[row, col] != 255:
            false_negatives += 1


# true-positives
true_positives=0
for row in range(total_rows):
    for col in range(total_cols):
        if ground_truth[row, col] == 2 and final[row, col] == 255:
            true_positives += 1


# true-negatives
true_negatives=0
for row in range(total_rows):
    for col in range(total_cols):
        if ground_truth[row, col] == 1 and final[row, col] != 255:
            true_positives += 1

total_points = total_cols*total_rows
assert(true_positives + true_negatives + false_positives + false_negatives == total_points)



# F
recall = true_positives / (true_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)
F = 2 * (precision * recall) / (precision + recall)

##################################
actual_positives = true_positives + false_negatives
actual_negatives = false_positives + true_negatives

#false-positive factor
false_positive_factor = false_positives/actual_positives

# fasle_negative_factor
false_negative_factor = false_negatives/actual_negatives

# true_positives to total points
true_positives_factor_total = true_positives/total_points

labels = ["F", "True positive"]
values = [F,  true_positives_factor_total]

plt.figure(figsize=(20, 10))
plt.bar(labels, values)
plt.savefig("2dOutput/shi_tomasi_1_Aloe_" + str(qualityLevel) + "_" + str(euclidesDistance) + "_" + str(1) + ".png")

labels2 = ["False negative", "False positive"]
values2 = [false_negative_factor, false_positive_factor]

plt.figure(figsize=(20, 10))
plt.bar(labels2, values2)
plt.savefig("2dOutput/shi_tomasi_2_Aloe_" + str(qualityLevel) + "_" + str(euclidesDistance) + "_" + str(1) + ".png")


