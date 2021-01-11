import numpy as np 
from common.constant import *
from common.clusteringAlgorithms import Semi_SupervisedClusteringAlgorithm
from utils.distance import MinkowskiDistance
from utils.data_load import softmax

# CE_sSC constants
class gammasIndex():
    selfinfo_gamma = 0  # 自身样本交叉熵惩罚系数
    pairinfo_gamma = 1  # 不同样本交叉熵惩罚系数
gammasIndex = gammasIndex()


class CE_sSCA(Semi_SupervisedClusteringAlgorithm):
    
    def __init__(self, X : np.ndarray, cluster_num : int, gammas = (0.01, 0.01)):
        super(CE_sSCA, self).__init__(X, cluster_num)

        # hyper parameter gammas
        self.gammas = gammas

        self.punish_matrix = np.eye(self.n) * gammas[gammasIndex.selfinfo_gamma]

    def inputSemi_SupervisedInformaintion(self, PairwiseConstraints : list, orderEmphasized = False):
        """
        根据输入的成对约束信息构建惩罚系数矩阵 \n
        :param PairwiseConstraints: 成对约束信息，一个元组列表 \n
        :param orderEmphasized: 是否强调成对约束顺序，若强调则 i 对 j 成对约束，但 j 对 i 不一定成对约束 \n
        元组数据结构为 (i : int , j : int , pair_type : bool) \n
        example-> [(0, 2, True), (2, 3, False)] 
        表示第一个样本对第三个样本成必连约束，第三个样本对第四个样本成不连约束 \n
        """

        for pairwise in set(PairwiseConstraints):
            i, j, pair_type = pairwise

            if pair_type:
                self.punish_matrix[i][j] = self.gammas[gammasIndex.pairinfo_gamma]
            else:
                self.punish_matrix[i][j] = - self.gammas[gammasIndex.pairinfo_gamma]

        if not orderEmphasized:
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        self.punish_matrix[i][j] += self.punish_matrix[j][i]

    def __Update_V__(self, U : np.ndarray):

        U_row_sum = np.sum(U, axis = 1)   # \sum_{j = 1}^{n} \mu_{ij} 
        # U_row_sum.shape = (c,)
        
        donation = np.zeros((self.n, self.c))
        
        for j in range(self.c):
            if U_row_sum[j] > epsilon:
                donation[:, j] = U.T[:, j] / U_row_sum[j]
        
        V = np.matmul(self.X, donation)
        return V
    
    def __Update_U__(self, V : np.ndarray, U : np.ndarray):
        new_U = np.zeros_like(U)

        for i in range(self.c):
            for j in range(self.n):
                new_U[i][j] = MinkowskiDistance(self.X[:, j], V[:, i]) ** 2
                
                # for rest two part
                _sum = 0.0
                for k in range(self.n):
                    if k != j:
                        if U[i][k] > epsilon:
                            _sum += self.punish_matrix[j][k] * math.log(U[i][k])
                        if U[i][j] > epsilon:
                            _sum += self.punish_matrix[k][j] * U[i][k] / U[i][j]
                
                # part add rest two part
                new_U[i][j] += _sum
                new_U[i][j] /= - self.punish_matrix[j][j]

        new_U = softmax(new_U)
        
        return new_U


    def iteration(self, initial_U : np.ndarray, iter_num = 200, quit_epsilon = epsilon):
        assert initial_U.shape[MatrixShapeIndex.row] == self.c and \
            initial_U.shape[MatrixShapeIndex.column] == self.n, "初始隶属度矩阵大小不匹配！"
        
        U = initial_U.copy()
        for t in range(iter_num):
            V = self.__Update_V__(U)
            U_save = U.copy()
            U = self.__Update_U__(V, U)

            f = Frobenius(U, U_save)
            if f < quit_epsilon:
                break
        
        return {
            'U' : U, 
            'V' : V, 
            't' : t, 
            'f' : f
        }
        



