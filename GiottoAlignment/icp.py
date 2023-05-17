import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


def icp(A, B, max_iterations=100, tolerance=1e-6):
    """
    Perform Iterative Closest Point (ICP) algorithm on two arrays of (n, 2) points
    """
    # Initialize transformation matrix to identity matrix
    T = np.eye(3)
    total_T = np.eye(3)
    plt.figure()

    
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
        
        # Use Singular Value Decomposition (SVD) to calculate rotation and translation matrices
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        t = centroid_B.T - np.dot(R, centroid_A.T)
        
        # Update transformation matrix
        T_new = np.eye(3)
        T_new[:2, :2] = R
        T_new[:2, 2] = t
        
        # Check for convergence
        if np.allclose(T, T_new, rtol=tolerance):
            break
        
        T = T_new


        total_T = np.dot(total_T, T_new)
        
        # Apply transformation to points A
        A = np.hstack((A, np.ones((len(A), 1))))
        A = np.dot(T, A.T).T[:, :2]

        plt.scatter(B[:, 0], -B[:, 1], c='b', label='B')
        plt.scatter(A[:, 0], -A[:, 1], c='r', label='A')
        if i == 0:  # only add legend in the first plot
            plt.legend()
        plt.title(f'Iteration {i+1}')
        plt.show()
        
    return total_T, i
