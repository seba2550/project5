import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """
        self.metric = metric


    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`

        Silhouette score is calculated as follows: For a single point i, take the smallest mean distance between i and all other points in any other cluster
        (inter-cluster distance) and subtract the mean distance between point i and all of the other points in its own cluster (intra-cluster distance). 
        Divide this by the max value of the two previous options. Score goes from -1 to 1. 
        """
        # Create two arrays: one will hold the scores and the other has euclidean distances for the X matrix
        scores = np.zeros(X.shape[0])
        distances = cdist(X, X, self.metric)

        for i in range(X.shape[0]):
            a = self.calculate_a(distances, y, i) 
            b = self.calculate_b(distances, y, i)
            scores[i] = (b - a)/np.max([a, b]) # Silhouette score for each observation in the X matrix

        self.scores = scores

        return self.scores

    def calculate_a(self, distances, labels, index):

        # Calculate the intra-cluster distance for each data point
        distances = distances[index,labels == labels[index]]
        a = np.sum(distances)/(np.sum(labels == labels[index]) - 1)
        
        return a

    def calculate_b(self, distances, labels, index):
    
        # Calculate the inter cluster distance for a data point
        inter_distances = np.ones(np.max(labels)) * np.inf
        for j in range(np.max(labels)):  # We're iterating over the assigned labels here
            if j != labels[index]:
                inter_distances[j] = np.sum(distances[index,labels == j])/np.sum(labels == j)
        b = np.min(inter_distances)
        
        return b