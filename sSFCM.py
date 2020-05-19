import numpy as np 
from common.constant import *
from common.clusteringAlgorithms import Semi_SupervisedWithPrioriKnowledgeClusteringAlgorithm
from utils.distance import get_EuclideanDistance_matrix
from utils.distance import MinkowskiDistance


class sSFCMA(Semi_SupervisedWithPrioriKnowledgeClusteringAlgorithm):

    def __init__(self, X : np.ndarray, cluster_num : int, m = 2):
        super(sSFCMA, self).__init__(X, cluster_num)
        
        # hyper parameter
        self.m = m

        if self.m == 1:
            self.__Update_U__ = self.__Update_U_for_m_equals_1__
        elif self.m > 1:
            self.__Update_U__ = self.__Update_U_for_m_greater_1__

    def __Update_U_for_m_equals_1__(self, V : np.ndarray, **kwarg) -> np.ndarray:
        
        U = self.tilde_U.copy()
        dis = np.zeros(self.c)
        for i in range(self.n):
            for j in range(self.c):
                dis[j] = MinkowskiDistance(self.X[:, i], V[:, j])
            l = np.argmin(dis)

            U[l][i] += 1 - np.sum(U[:, i])
        return U

    def __Update_U_for_m_greater_1__(self, V : np.ndarray, **kwarg) -> np.ndarray:
        
        if 'U' not in kwarg.keys():
            U = np.zeros((self.c, self.n))
        else:
            U = kwarg['U']
        
        times = 1 / (self.m - 1)
        distance_matrix = get_EuclideanDistance_matrix(self.X, V)
        coefficient_array = 1 - np.sum(self.tilde_U, axis = 0)
        
        for i in range(self.c):
            for j in range(self.n):
                if distance_matrix[j][i] > epsilon:
                    _sum = 0
                    for h in range(self.c):
                        if distance_matrix[j][h] > epsilon:
                            _sum += (1 / distance_matrix[j][h] ** 2) ** times
                    
                    U[i][j] = self.tilde_U[i][j] + coefficient_array[j] / _sum * ((1 / distance_matrix[j][i] ** 2) ** times)
                else:
                    U[i][j] = 1
        return U    
    
    def __Update_V__(self, U : np.ndarray) -> np.ndarray:

        U_power_m = U ** self.m
        U_row_sum = np.sum(U_power_m, axis=1)   # \sum_{j = 1}^{n} \mu_{ij} 
        # U_row_sum.shape = (c,)
        
        donation = np.zeros((self.n, self.c))
        
        for j in range(self.c):
            if U_row_sum[j] > epsilon:
                donation[:, j] = U_power_m.T[:, j] / U_row_sum[j]
        
        V = np.matmul(self.X, donation)
        return V

    def iteration(self, iter_num = 200, quit_epsilon = epsilon):
        import time
        start = time.clock()

        V = self.initial_V.copy()
        U = self.__Update_U__(V)

        for t in range(iter_num):
            V_save = V.copy()
            V = self.__Update_V__(U)
            
            U_save = U.copy()
            U = self.__Update_U__(V, U = U)
            
            V_f = Frobenius(V, V_save) 
            if V_f < quit_epsilon:
                break
            
            U_f = Frobenius(U, U_save) 
            if U_f < quit_epsilon:
                break

        elapsed = (time.clock() - start)

        return {
            'U' : U, 
            'V' : V, 
            't' : t, 
            'U_f' : U_f,
            'V_f' : V_f,
            'use_time' : elapsed
        }
