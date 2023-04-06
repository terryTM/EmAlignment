"""
# load image
img = cv2.imread("/Users/terryma/Documents/emshot1.png")

# convert to graky
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold input image as mask
mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]

# negate mask
mask = 255 - mask

# apply morphology to remove isolated extraneous noise
# use borderconstant of black since foreground touches the edges
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# anti-alias the mask -- blur then stretch
# blur alpha channel
mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

# linear stretch so that 127.5 goes to 0, but 255 stays 255
mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

# put mask into alpha channel
result = img.copy()
result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
result[:, :, 3] = mask

# display result, though it won't show transparency
cv2.imshow("MASK", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


import cv2
import numpy as np
import pyntcloud
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.spatial.transform import Rotation as R
import pandas as pd
import math
from icp import icp
import matplotlib.pyplot as plt
#from simpleicp import PointCloud, SimpleICP




img1 = cv2.imread("/Users/terryma/Documents/GiottoAlignment/emshot1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/Users/terryma/Documents/GiottoAlignment/emshot2.png", cv2.IMREAD_GRAYSCALE)
   
def imageContour(path):   
    # Load the image as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    img = cv2.equalizeHist(img)
    img[img > 100] = 255
    img = cv2.medianBlur(img, 15)

    img[img < 100] = 0

    # Threshold the image to obtain a binary mask
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)

    # Find the contours of all objects in the image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    return largest_contour


largest_contour1 = imageContour("/Users/terryma/Documents/GiottoAlignment/emshot1.png")
largest_contour2 = imageContour("/Users/terryma/Documents/GiottoAlignment/emshot2.png")


# Convert contour to point cloud
def contourToPointcloud(contour):
    # Reshape contour to (n_points, 1, 2)
    points = contour.reshape(-1, 1, 2)
    # Stack (x, y) coordinates vertically to create 2D array
    pointcloud = np.vstack(points[:, 0])
    return pointcloud

pointcloud1 = contourToPointcloud(largest_contour1)
pointcloud2 = contourToPointcloud(largest_contour2)
print(pointcloud1.shape)
print(pointcloud2.shape)
A = pointcloud1
B = pointcloud2


# Apply ICP to align A to B
T = icp(A, B)

# Apply transformation to A and plot the aligned point clouds
A_aligned = np.hstack((A, np.ones((len(A), 1))))
A_aligned = np.dot(T, A_aligned.T).T[:, :2]

import matplotlib.pyplot as plt

plt.scatter(B[:, 0], -B[:, 1], c='b', label='B')
plt.scatter(A[:, 0], -A[:, 1], c='r', label='A')
plt.scatter(A_aligned[:, 0], -A_aligned[:, 1], c='g', label='A aligned')
plt.legend()
plt.show()


'''
print(pointcloud1.shape)
x = pointcloud2[:, 0]
y = -pointcloud2[:, 1]
plt.scatter(x, y)
plt.show()

cv2.imshow("align", pointcloud1)
cv2.waitKey(0)
'''