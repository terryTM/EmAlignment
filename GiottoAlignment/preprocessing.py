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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.spatial.transform import Rotation as R
import pandas as pd
import math
from icp import icp
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN


#from simpleicp import PointCloud, SimpleICP



   
def preprocess(path):
    img7 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img7 = cv2.threshold(img7, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    img7[img7 < 250] = 0

    img7 = cv2.GaussianBlur(img7, (5,5), 0)

    img7 = cv2.medianBlur(img7,3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    img7 = cv2.morphologyEx(img7, cv2.MORPH_BLACKHAT, kernel)

    img7 = cv2.medianBlur(img7,15)


    return img7


def imageContour(img):   
    # Load the image as grayscale
    img = cv2.equalizeHist(img)
    img[img > 50] = 255

    kernel = np.ones(7, np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)



    #img[img < 100] = 0

    # Threshold the image to obtain a binary mask
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)

    mask = (255-mask)

    return mask



    # Find the contours of all objects in the image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    



    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    cim = np.zeros_like(img)

    cv2.drawContours(cim, largest_contour, -1, 255, 1)
    cv2.imshow("con", cim)
    cv2.waitKey(0)
    return largest_contour



process_img1 = preprocess("/Users/terryma/Documents/GiottoAlignment/whole_embryo_H3K4me3_E11_50um.png")
process_img2 = preprocess("/Users/terryma/Documents/GiottoAlignment/whole_embryo_H3K27me3_E11_50um.png")



largest_contour1 = imageContour(process_img1)

#cv2.imshow("con", largest_contour1)
#cv2.waitKey(0)


largest_contour2 = imageContour(process_img2)


# Convert contour to point cloud
#def contourToPointcloud(contour):
#    # Reshape contour to (n_points, 1, 2)
#    points = contour.reshape(-1, 1, 2)
#    # Stack (x, y) coordinates vertically to create 2D array
#    pointcloud = np.vstack(points[:, 0])
#    return pointcloud

def image_to_pointcloud(img):
    # Get the indices of the non-zero pixels
    points = np.nonzero(img)
    height = img.shape[0]
    # Flip the y-coordinates
    points_flipped = (points[1], height - points[0])
    # Stack the indices to create the point cloud
    pointcloud = np.column_stack(points_flipped)
    return pointcloud


def downsample(pointcloud, n_clusters):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=6 * n_clusters)
    kmeans = kmeans.partial_fit(pointcloud)
    return kmeans.cluster_centers_



def dbscan_outlier_removal(point_cloud, eps=80, min_samples=10):
    """
    Remove outliers from a point cloud using DBSCAN clustering.
    
    point_cloud: input point cloud as a numpy array.
    eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    """
    # Apply DBSCAN on the point cloud.
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(point_cloud)

    # Get the indices of the core samples.
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Create a mask to filter out the outliers.
    mask = core_samples_mask

    # Apply the mask to the point cloud.
    filtered_point_cloud = point_cloud[mask]

    return filtered_point_cloud



pointcloud1 = image_to_pointcloud(largest_contour1)
pointcloud2 = image_to_pointcloud(largest_contour2)

pointcloud1 = downsample(pointcloud1, 2000)
pointcloud2 = downsample(pointcloud2, 2000)

pointcloud1 = dbscan_outlier_removal(pointcloud1)
pointcloud2 = dbscan_outlier_removal(pointcloud2)

pointcloud1[:,1] = pointcloud1[:,1] * -1
pointcloud2[:,1] = pointcloud2[:,1] * -1


print(pointcloud1.shape)
print(pointcloud2.shape)
A = pointcloud1
B = pointcloud2


# Apply ICP to align A to B
T, i = icp(A, B)

#| a  b  tx |
#| c  d  ty |
#| 0  0  1  |

print(T)

# Apply transformation to points A
A = np.hstack((A, np.ones((len(A), 1))))
A = np.dot(T, A.T).T[:, :2]

plt.scatter(B[:, 0], -B[:, 1], c='g', label='B')
plt.scatter(A[:, 0], -A[:, 1], c='r', label='A')
plt.legend()
plt.title(f'FINAL Iteration {i+1}')
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