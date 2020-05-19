import math
import numpy as np 
from common.constant import *
from common.clusteringAlgorithms import Semi_SupervisedWithPrioriKnowledgeClusteringAlgorithm
from utils.distance import Mahalanobis_distance
from utils.data_load import normalize

class SMUCA(Semi_SupervisedWithPrioriKnowledgeClusteringAlgorithm):
    def __init__(self, X : np.ndarray, cluster_num : int, gamma = 0.01):
        super(SMUCA, self).__init__(X, cluster_num)
        
        # hyper parameter
        self.gamma =  gamma

    def __get_initial_V__(self) -> np.ndarray:

        if self.gave_tilde_U:
            U_power_m = self.tilde_U ** 2
            U_row_sum = np.sum(U_power_m, axis = 1)   # \sum_{j = 1}^{n} \mu_{ij} 
            # U_row_sum.shape = (c,)
            
            donation = np.zeros((self.n, self.c))
            
            for j in range(self.c):
                if U_row_sum[j] > epsilon:
                    donation[:, j] = U_power_m.T[:, j] / U_row_sum[j]
            
            V = np.matmul(self.X, donation)
        
        else:   # 如果没有先验隶属度信息，则随机初始化初始聚类中心矩阵
            V = np.random.rand(self.feature_num, self.c)
        return V

    def __get_C_matrix__(self, initial_V : np.ndarray):
        c = self.c
        n = self.n
        feature_num = self.feature_num

        C = np.zeros((feature_num, feature_num))
        for i in range(n):
            for k in range(c):
                temp_vector = (self.X[:, i] - initial_V[:, k]).reshape((-1, 1))
                C += (self.tilde_U[k][i] ** 2) * (temp_vector * temp_vector.T)
        C /= n
        
        # 若行列式绝对值小于epsilon 认为是奇异阵，加单位阵来近似
        if abs(np.linalg.det(C)) < epsilon:
            C += np.eye(self.feature_num)

        return C

    def __get_Mahalanobis_distance_matrix__(self, V : np.ndarray, A : np.ndarray) -> np.ndarray:
        c = self.c
        n = self.n

        distance_matrix = np.zeros((n, c))

        for i in range(n):
            for j in range(c):
                distance_matrix[i][j] = Mahalanobis_distance(self.X[:, i], V[:, j], A)
        return distance_matrix

    def __Update_V__(self, U : np.ndarray) -> np.ndarray:
        n, c = self.n, self.c

        U_row_sum = np.sum(U, axis=1)   # \sum_{j = 1}^{n} \mu_{ij} 
        # U_row_sum.shape = (c,)
        
        donation = np.zeros((n, c))
        
        for j in range(c):
            if U_row_sum[j] > epsilon:
                donation[:, j] = U.T[:, j] / U_row_sum[j]
        
        V = np.matmul(self.X, donation)
        return V

    def __Updata_U__(self, U : np.ndarray, V : np.ndarray, A : np.ndarray) -> np.ndarray:
        n, c = self.n, self.c

        coefficient_array = 1 - np.sum(self.tilde_U, axis = 0)   # 1 - \sum_{h - 1}^c \tilde{u}_ij, shape = (n,)
        distance_matrix = self.__get_Mahalanobis_distance_matrix__(V, A) ** 2

        for i in range(c):
            for j in range(n):
                U[i][j] = math.exp(- distance_matrix[j][i] / self.gamma)

        U = normalize(U)
        U *= coefficient_array

        U += self.tilde_U

        return U

    def iteration(self, iter_num = 200, quit_epsilon = epsilon):
        import time
        start = time.clock()

        c, n = self.c, self.n
        U = np.zeros((c, n))
        # V = self.__get_initial_V__()  # 论文中用先验隶属矩阵生成初始聚类中心
        V = self.initial_V.copy()
        C = self.__get_C_matrix__(V)

        # 计算协方差矩阵C的逆矩阵，如果C为奇异阵，则加个单位矩阵再求逆阵，否则直接求
        try:
            A = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            A = np.linalg.inv(C + np.eye(self.feature_num))

        # A = np.linalg.pinv(C)   # 计算 C 的广义逆矩阵

        for t in range(iter_num):
            U_save = U.copy()
            U = self.__Updata_U__(U, V, A)
            # print(U)
            
            V_save = V.copy()
            V = self.__Update_V__(U)
            
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