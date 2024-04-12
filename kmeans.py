"""
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
"""

import numpy as np


class KMeans(object):
    def __init__(self, points, k, init="random", max_iters=10000, rel_tol=1e-05):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == "random":
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        indices = np.random.choice(self.points.shape[0], self.K, replace=False)
        self.centers = self.points[indices]
        return self.centers
        
    def kmpp_init(self):
        """
            Use the intuition that points further away from each other will probably be better initial centers
        Return:
            self.centers : K x D numpy array, the centers.
        """
        sample_size = int(0.01 * len(self.points))
        sampled_points = self.points[np.random.choice(self.points.shape[0], size=sample_size, replace=False)]

        centers = [sampled_points[np.random.randint(sampled_points.shape[0])]]
        for _ in range(1, self.K):
            dist = np.array([pairwise_dist(sampled_points,center) for center in centers])

            min_dist = np.min(dist, axis=0)
            next_center = sampled_points[np.argmax(min_dist)]
            centers.append(next_center)
        self.centers = np.array(centers)

    def update_assignment(self):
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: Donot use loops for the update_assignment function
        Hint: You could call pairwise_dist() function
        Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison.
        """
        try:
            dist = pairwise_dist(self.points, self.centers)
        except:
            dist=np.linalg.norm(self.points[:, np.newaxis] - self.centers, axis=2)**2
        self.assignments = np.argmin(dist, axis=1)
        return self.assignments

    def update_centers(self):
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        HINT: If there is an empty cluster then it won't have a cluster center, in that case the number of rows in self.centers can be less than K.
        """
        new_centers = []
        for i in range(self.K):
            cluster_points = self.points[self.assignments == i]
            if len(cluster_points) > 0:
                new_center = np.mean(cluster_points, axis=0)
                new_centers.append(new_center)
        self.centers = np.array(new_centers, dtype=float)
        return self.centers

    def get_loss(self):
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        dist = np.linalg.norm(self.points - self.centers[self.assignments], axis=1)**2
        self.loss = np.sum(dist)
        return self.loss

    def train(self):
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster,
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned,
                     pick a random point in the dataset to be the new center and
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference
                     in loss compared to the previous iteration is less than the given
                     relative tolerance threshold (self.rel_tol).
                   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!

        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.

        HINT: Donot loop over all the points in every iteration. This may result in time out errors
        HINT: Make sure to care of empty clusters. If there is an empty cluster the number of rows in self.centers can be less than K.
        """
        prev_loss = np.inf
        # Iterate max_iters time
        for i in range(self.max_iters):
            #update assignment
            self.update_assignment()
            #update cluster center
            self.update_centers()
            #check if there is a cluster without mean
            for j in range(len(self.centers)):  
                if j not in self.assignments:
                    random_index = np.random.choice(self.points.shape[0])
                    self.centers[j] = self.points[random_index]
                    self.update_assignment()  
            #calculate the loss
            curr_loss = self.get_loss()
            #If converges
            if np.abs(prev_loss - curr_loss) / prev_loss < self.rel_tol:
                break
            prev_loss = curr_loss

        return self.centers, self.assignments, self.loss


def pairwise_dist(x, y):
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between
            x[i, :] and y[j, :]

    HINT: Do not use loops for the pairwise_distance function
    """
    sum_x_square = np.sum(np.square(x),axis=1,keepdims=True)
    sum_y_square = np.sum(np.square(y), axis=1)
    xy_product = np.dot(x, y.T)
    return np.sqrt(sum_x_square + sum_y_square - 2 * xy_product)

def rand_statistic(xGroundTruth, xPredicted):
    """
    Args:
        xPredicted : list of length N where N = no. of test samples
        xGroundTruth: list of length N where N = no. of test samples
    Return:
        Rand statistic value: final coefficient value as a float

    HINT: You can use loops for this function.
    HINT: The idea is to make the comparison of Predicted and Ground truth in pairs.
        1. Choose a pair of points from the Prediction.
        2. Compare the prediction pair pattern with the ground truth pair.
        3. Based on the analysis, we can figure out whether it's a TP/FP/FN/FP.
        4. Then calculate rand statistic value
    """
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    N = len(xGroundTruth)
    for i in range(N):
        for j in range(i+1, N):
            if (xGroundTruth[i] == xGroundTruth[j]) and (xPredicted[i] == xPredicted[j]):
                TP += 1
            elif (xGroundTruth[i] == xGroundTruth[j]) and (xPredicted[i] != xPredicted[j]):
                FN += 1
            elif (xGroundTruth[i] != xGroundTruth[j]) and (xPredicted[i] != xPredicted[j]):
                TN += 1
            else:
                FP += 1 
    return (TP + TN) / (TP + FP + FN + TN)

