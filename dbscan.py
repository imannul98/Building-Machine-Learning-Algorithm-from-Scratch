import numpy as np
from kmeans import pairwise_dist


class DBSCAN(object):
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset

    def fit(self):
        """
        Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        See in what order the clusters are being expanded and new points are being checked, recommended to check neighbors of a point then decide whether to expand the cluster or not.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """
        C=0
        n_points = len(self.dataset)
        visitedIndex=set()
        cluster_idx=np.zeros(n_points)-1
        #for each unvisited point P in dataset X
        for P in range(n_points):
            #mark P as visited 
            if P not in visitedIndex:
                visitedIndex.add(P)
                neighborpts=self.regionQuery(P)
                if len(neighborpts)<self.minPts:
                    cluster_idx[P]=-1
                else:
                    self.expandCluster(P,neighborpts,C,cluster_idx,visitedIndex)
                    C+=1
        return cluster_idx





    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """
        Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints:
            1. np.concatenate(), and np.sort() may be helpful here. A while loop may be better than a for loop.
            2. Use, np.unique(), np.take() to ensure that you don't re-explore the same Indices. This way we avoid redundancy.
        """
        #add P to cluster C
        neighborIndices=np.sort(neighborIndices)
        neighborIndices_unique=np.copy(neighborIndices)
        cluster_idx[index]=C
        
        i=0
        #for each point P' in NeighborPts
        while i<len(neighborIndices):
            #if P' is not visited
            if neighborIndices[i] not in visitedIndices:
                #mark P' as visited
                visitedIndices.add(neighborIndices[i])
                visited_array=np.array(list(visitedIndices))
                #NeighborPts' = regionQuery(P', eps)
                neighbor_point_member=self.regionQuery(neighborIndices[i])
                #if sizeof(NeighborPts') >= MinPts 
                if len(neighbor_point_member)>=self.minPts:
                    #NeighborPts = NeighborPts joined with NeighborPts‘
                    neighborIndices = np.concatenate((neighborIndices, neighbor_point_member))
                    unique_elements, indices = np.unique(neighborIndices, return_index=True)
                    sorted_indices = np.sort(indices)
                    neighborIndices = np.take(neighborIndices, sorted_indices)
                
            #if P' is not yet member of any cluster
            if cluster_idx[neighborIndices[i]]<0:
                cluster_idx[neighborIndices[i]]=C
            i+=1


        

    def regionQuery(self, pointIndex):
        """
        Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        distances=pairwise_dist(self.dataset,self.dataset[pointIndex].reshape(1, -1))
        indices=np.argwhere(distances < self.eps)[:,0]
        return indices

