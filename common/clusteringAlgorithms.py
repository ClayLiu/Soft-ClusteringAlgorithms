import numpy as np 
from common.constant import *
from utils.data_load import normalize

class ClusteringAlgorithm():
    """ 聚类算法 """
    def __init__(self, X : np.ndarray, cluster_num : int):
        self.X = X
        
        self.c = cluster_num
        self.n = X.shape[MatrixShapeIndex.column]
        self.feature_num = X.shape[MatrixShapeIndex.row]

    def __get_initial_U__(self) -> np.ndarray:
        U = np.random.rand(self.c, self.n)
        U = normalize(U)
        return U

    def __Update_U__(self, V : np.ndarray, U : np.ndarray) -> np.ndarray:
        # U for data input
        pass

    def __Update_V__(self, U : np.ndarray) -> np.ndarray:
        pass

    def iteration(self, iter_num = 200, quit_epsilon = epsilon):
        import time
        start = time.clock()

        random_choice = np.random.randint(0, self.n, self.c)
        initial_V = self.X[:, random_choice]
        V = initial_V.copy()
        U = self.__get_initial_U__()
        
        for t in range(iter_num):
            U_save = U.copy()    
            U = self.__Update_U__(V, U)
            U_f = Frobenius(U, U_save) 
            if U_f < quit_epsilon:
                break
            
            V_save = V.copy()
            V = self.__Update_V__(U)
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

class Semi_SupervisedClusteringAlgorithm(ClusteringAlgorithm):
    """ 半监督聚类算法 """
    def inputSemi_SupervisedInformaintion(self):
        pass

class Semi_SupervisedWithPrioriKnowledgeClusteringAlgorithm(Semi_SupervisedClusteringAlgorithm):
    """ 以先验知识半监督的聚类算法 """
    
    def __init__(self, X : np.ndarray, cluster_num : int):
        super(Semi_SupervisedWithPrioriKnowledgeClusteringAlgorithm, self).__init__(X, cluster_num)
        self.gave_tilde_U = False
        self.tilde_U = np.zeros((self.c, self.n))
        self.initial_V = np.random.rand(self.feature_num, self.c)

    def inputSemi_SupervisedInformaintion(self, tilde_U : np.ndarray, initial_V : np.ndarray):
        assert tilde_U.shape[MatrixShapeIndex.row] == self.c and \
            tilde_U.shape[MatrixShapeIndex.column] == self.n, '先验隶属度矩阵大小不匹配！'

        assert initial_V.shape[MatrixShapeIndex.column] == self.c and \
            initial_V.shape[MatrixShapeIndex.row] == self.feature_num, '初始聚类中心矩阵大小不匹配！' 

        self.tilde_U = tilde_U.copy()
        self.initial_V = initial_V
        self.gave_tilde_U = True

