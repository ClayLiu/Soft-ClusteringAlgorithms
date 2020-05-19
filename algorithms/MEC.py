import numpy as np 
from common.constant import *
from common.clusteringAlgorithms import ClusteringAlgorithm
from utils.data_load import softmax
from utils.distance import get_EuclideanDistance_matrix


class MECA(ClusteringAlgorithm):
    
    def __init__(self, X : np.ndarray, cluster_num : int, gamma = 0.01):
        super(MECA, self).__init__(X, cluster_num)
        
        # hyper parameter
        self.gamma =  gamma
    
    def __Update_U__(self, V :np.ndarray, U : np.ndarray) -> np.ndarray:

        distance_matrix = get_EuclideanDistance_matrix(V, self.X) ** 2
        distance_matrix /= - self.gamma
        U = softmax(distance_matrix)

        return U
     
    def __Update_V__(self, U : np.ndarray):

        U_row_sum = np.sum(U, axis = 1)   # \sum_{j = 1}^{n} \mu_{ij} 
        # U_row_sum.shape = (c,)
        
        donation = np.zeros((self.n, self.c))
        
        for j in range(self.c):
            if U_row_sum[j] > epsilon:
                donation[:, j] = U.T[:, j] / U_row_sum[j]
        
        V = np.matmul(self.X, donation)
        return V


