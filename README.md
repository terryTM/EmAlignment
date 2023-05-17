# Embryo Alignment

Iterative Closest Point alignment method for two mouse embryos. 

### Methodology
Preprocessing is conduced using filtering and thresholding using OpenCV. Then we convert the image to a point cloud and use kmeans to sample 2000 points. We then use DBSCAN clustering to remove outliers by defining maximum distance between two samples and minimum number of samples in a neighborhood. Finally we perform ICP until convergence.

### Required Packages
- OpenCV
- Numpy
- sklearn
- scipy
- pandas
- matplotlib
