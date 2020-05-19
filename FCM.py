import numpy as np 
from common.constant import *
from common.clusteringAlgorithms import ClusteringAlgorithm
from utils.distance import get_EuclideanDistance_matrix

class FCMA(ClusteringAlgorithm):

    def __init__(self, X : np.ndarray, cluster_num : int, m = 2):
        super(FCMA, self).__init__(X, cluster_num)
        
        # hyper parameter
        self.m = m
        

    def __Update_U__(self, V : np.ndarray, U : np.ndarray) -> np.ndarray:
        distance_matrix = get_EuclideanDistance_matrix(self.X, V)
                
        times = 2 / (self.m - 1)

        for i in range(self.c):
            for j in range(self.n):
                if distance_matrix[j][i] > epsilon: # 如果 x_j 离 v_i 太近的话，那 x_j 直接看作属于 i 类
                    _sum = 0
                    for h in range(self.c):
                        if distance_matrix[j][h] > epsilon: # 如果 x_j 离 v_h 太近的话，那 x_h 可以直接看作属于 h 类的聚类中心
                            _sum += (distance_matrix[j][i] / distance_matrix[j][h]) ** times
                    U[i][j] = 1 / _sum
                else:
                    U[i][j] = 1
        return U

    def __Update_V__(self, U : np.ndarray) -> np.ndarray:

        U_power_m = U ** self.m
        U_row_sum = np.sum(U_power_m, axis = 1)   # \sum_{j = 1}^{n} \mu_{ij} 
        # U_row_sum.shape = (c,)
        
        donation = np.zeros((self.n, self.c))
        
        for j in range(self.c):
            if U_row_sum[j] > epsilon:
                donation[:, j] = U_power_m.T[:, j] / U_row_sum[j]
        
        V = np.matmul(self.X, donation)
        return V
