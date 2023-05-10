import numpy as np
from scipy.spatial import KDTree

def icp(A, B, max_iterations=100, tolerance=1e-6):
    """
    Perform Iterative Closest Point (ICP) algorithm on two arrays of (n, 2) points
    """
    # Initialize transformation matrix to identity matrix
    T = np.eye(3)
    
    # Perform ICP for max_iterations
    for i in range(max_iterations):
        # Find the nearest neighbors of each point in A in B
        tree = KDTree(B)
        distances, indices = tree.query(A)
        closest_points = B[indices]
        
        # Calculate the centroid of each set of points
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(closest_points, axis=0)
        
        # Calculate the covariance matrix of the points
        H = np.dot((A - centroid_A).T, closest_points - centroid_B)
        
        # Check for convergence
        if np.allclose(T, T_new, rtol=tolerance):
            break
        
        T = T_new
        
        # Apply transformation to points A
        A = np.hstack((A, np.ones((len(A), 1))))
        A = np.dot(T, A.T).T[:, :2]
        
    return T
