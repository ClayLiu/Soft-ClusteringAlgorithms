''' sFCM_HC.py '''
import numpy as np 
from common.constant import *
from common.clusteringAlgorithms import Semi_SupervisedWithPrioriKnowledgeClusteringAlgorithm
from utils.data_load import normalize
from utils.distance import MinkowskiDistance


class sFCM_HCA(Semi_SupervisedWithPrioriKnowledgeClusteringAlgorithm):
    
    def __init__(self, X : np.ndarray, cluster_num : int, gamma = 0.01, m = 2):
        super(sFCM_HCA, self).__init__(X, cluster_num)
        
        # hyper parameter
        self.gamma = gamma
        self.m = m
        

    def __get_initial_U__(self, initial_V : np.ndarray):
        U = np.zeros((self.c, self.n))
        return self.__Update_U_for_m_eq_1__(initial_V, U)


    def __Update_U__(self, V : np.ndarray, U : np.ndarray) -> np.ndarray:
        if self.m == 1:
            return self.__Update_U_for_m_eq_1__(V, U)
        else:
            return self.__Update_U_for_any_m__(V, U)


    def __Update_U_for_m_eq_1__(self, V : np.ndarray, U : np.ndarray):
        for i in range(self.c):
            for j in range(self.n):
                dis = MinkowskiDistance(self.X[:, j], V[:, i])
                
                if self.tilde_U[i][j] < epsilon:
                    U[i][j] = math.exp(- dis ** 2 / self.gamma)
                else:
                    U[i][j] = self.tilde_U[i][j] * math.exp(- dis ** 2 / self.gamma)

        U = normalize(U)
        return U 

    def __Update_U_for_any_m__(self, V : np.ndarray, U : np.ndarray):
        
        new_U = np.zeros_like(U)    # 2020年4月24日22:40:11 之前没有加这个，导致 U 迭代有问题，然而结果还是那样

        for i in range(self.c):
            for j in range(self.n):
                dis = MinkowskiDistance(self.X[:, j], V[:, i]) ** 2
                
                if self.tilde_U[i][j] < epsilon:
                    new_U[i][j] = math.exp(-  self.m * (U[i][j] ** (self.m - 1)) * dis / self.gamma)
                else:
                    new_U[i][j] = self.tilde_U[i][j] * math.exp(- self.m * (U[i][j] ** (self.m - 1)) * dis / self.gamma)

        new_U = normalize(new_U)
        
        return new_U 


    def __Update_V__(self, U : np.ndarray):

        U_power_m = U ** self.m
        U_row_sum = np.sum(U_power_m, axis = 1)   # \sum_{j = 1}^{n} \mu_{ij} 
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
        U = self.__get_initial_U__(V)

        for t in range(iter_num):
            V_save = V.copy()
            V = self.__Update_V__(U)

            U_save = U.copy()    
            U = self.__Update_U__(V, U)
            
            U_f = Frobenius(U, U_save) 
            if U_f < quit_epsilon:
                break

            V_f = Frobenius(V, V_save) 
            if V_f < quit_epsilon:
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
