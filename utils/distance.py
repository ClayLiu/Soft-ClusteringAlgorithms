import math
import numpy as np 
from common.constant import *

def MinkowskiDistance(vector_1 : np.array, vector_2 : np.array, p = 2) -> float:
    """
    计算两点的闵可夫斯基距离 \n
    p = 1 时为曼哈顿距离 \n
    p = 2 时为欧式距离 \n
    p = ∞ 时为切比雪夫距离 \n
    """
    
    temp = np.abs(vector_1 - vector_2)
    return np.sum(temp ** p) ** (1 / p)

def Mahalanobis_distance(vector_1 : np.ndarray, vector_2 : np.ndarray, S_inv : np.ndarray) -> float:
    """
    计算两点的马哈拉诺比斯距离 \n
    :param S_inv: 协方差矩阵的逆矩阵 \n
    """
    temp_vector = vector_1 - vector_2
    temp_vector = temp_vector.reshape((-1, 1))
    
    dot_product = np.dot(temp_vector.T, np.dot(S_inv, temp_vector))    
    if dot_product < 0:
        raise Exception('The number to calculate the square root is negative.')
    dis = math.sqrt(dot_product)
    return dis

def get_EuclideanDistance_matrix(X : np.ndarray, V : np.ndarray) -> np.ndarray:
    c = V.shape[MatrixShapeIndex.column]
    n = X.shape[MatrixShapeIndex.column]

    distance_matrix = np.zeros((n, c))
    
    for i in range(n):
        for j in range(c):
            distance_matrix[i][j] = MinkowskiDistance(X[:, i], V[:, j])
    
    return distance_matrix

if __name__ == '__main__':
    a = np.array([1, 2])
    b = np.array([4, 6])
    print(MinkowskiDistance(a, b, 1))