import cv2
import json
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import sys

if len(sys.argv) != 3:
    print("Wrong number of arguments")
    exit(1)

# left_img = "CrackForest/image/001.jpg"
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

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create(threshold=25)

# find and draw the keypoints
kp = fast.detect(img1,None)
ground_truth = np.ones(shape=(img1.shape))

for point in kp:
	x = int(kp[0].pt[0])
	y = int(kp[0].pt[1])
	ground_truth[y,x] = 2

total_cols = len(img1)
total_rows = len(img1[0])

k = 0.04

blockSize = int(sys.argv[1])
ksize = int(sys.argv[2])

dst = cv2.cornerHarris(img1, blockSize, ksize, k)
ret, final = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)

corner_value = 0
corners = []

# high is y, width is x
for i in range(len(final)):
    for j in range(len(final[0])):
        if corner_value == final[i, j]:
            corners.append([j, i])

#visualise ground-truth
total_cols = len(ground_truth[0])
total_rows = len(ground_truth)

black = 255
white = 0

ground_truth_v = np.zeros((total_rows, total_cols), np.uint8)


for row in range(total_rows):
    for col in range(total_cols):
        if ground_truth[row,col] == 1:
            ground_truth_v[row, col] = white
        elif ground_truth[row,col] == 2:
            ground_truth_v[row,col] = black

plt.imshow(ground_truth_v, cmap="gray")

#compare with our detector
plt.imshow(dst, cmap="gray")
print(np.max(final))
print(np.min(final))
print(np.average(final))
print(np.histogram(final))

plt.imshow(final, cmap="gray")

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

false_positive_factor = false_positives/actual_positives

false_negative_factor = false_negatives/actual_negatives

true_positives_factor_total = true_positives/total_points

labels = ["F", "False negative", "False positive", "True positive"]
values = [F, false_negative_factor, false_positive_factor, true_positives_factor_total]
plt.figure(figsize=(20, 10))
plt.bar(labels, values)
plt.savefig("2dOutput/harrisAloe_" + str(blockSize) + "_" + str(ksize) + ".png")


