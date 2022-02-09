import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if k <= 0:
            raise ValueError("K must be greater than 0!")
        # Pass the input variables as attributes of the KMeans class
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric
        self.error = np.inf

        # Build empty lists for holding the clusters and centroids
        self.clusters = [[] for _ in range(self.k) ] # Holds indices
        self.centroids = [] # Holds samples



    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # Get the dimensions of the input feature matrix
        self.mat = mat
        self.n_observations, self.n_features = mat.shape

        
        # Initialize random centroids and create an object to hold previous centroids as we update them through the fitting process
        centroids = np.random.rand(self.k, self.n_features)
        prev_centroids = centroids.copy()

        # This loop continues until one of two events: we reach the user-input maximum iterations for the fitting process, or we reach convergence (i.e.: the MSE for the current centroids and the previous centroids is within the user-input tolerable threshold)
        for i in range(self.max_iter):

            dist = np.linalg.norm(self.mat - centroids[0,:],axis=1).reshape(-1,1)
            for classification in range(1,self.k):
                dist = np.append(dist,np.linalg.norm(self.mat - centroids[classification,:],axis=1).reshape(-1,1),axis=1) # Calculate the Euclidean distance from each point to each centroid
            
            classes = np.argmin(dist,axis=1)  # Class label will be the minimum distance here. In other words, the centroid that the point is closest to
            
            # Update centroid positions
            for classification in set(classes):
                centroids[classification,:] = np.mean(self.mat[classes == classification,:],axis=0) # Mean position of the cluster gives the new centroid

            current_error = np.mean(np.square(dist))


            if np.isclose(self.error, current_error, self.tol): # If the MSEs for the current and previous centroids (respectively) are within the tolerable threshold, we've reached convergence and break the loop
                self.error = current_error # Store the current MSE as an attribute for the get_error method
                break

            self.error = current_error # Store the current MSE so that we can compare it to the MSE for the next centroids in the following interation. This way we also get an MSE for a model that reaches the maximum number of iterations


        self.centroids = centroids # Store the current centroids so that they can be accessed with the get_centroids method, as well as for the predict method

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        dist = np.linalg.norm(self.mat - self.centroids[0,:], axis = 1).reshape(-1,1) # Once again calculate euclidean distance from each point to each centroid
        for classification in range(1, self.k):
            dist = np.append(dist, np.linalg.norm(self.mat - self.centroids[classification, :], axis = 1).reshape(-1,1), axis = 1) 

        self.classes = np.argmin(dist, axis = 1) # Take the index of the minimum distance in the dist array. This will be the classification/label for the data point

        return self.classes # Return the labels

    # Last two methods are pretty self-explanatory (no pun intended), they simply return the final MSE and centroids for the trained model
    def get_error(self) -> float:
        """
        returns the final mean-squared error of the fit model

        outputs:
            float
                the mean-squared error of the fit model
        """
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids