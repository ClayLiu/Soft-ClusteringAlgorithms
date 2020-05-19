import math
import numpy as np 

class Bounds():
    upper = 1
    lower = 0

class MatrixShapeIndex():
    row = 0
    column = 1

MatrixShapeIndex = MatrixShapeIndex()
Bounds = Bounds()

epsilon = 1e-5

def Frobenius(matrix : np.ndarray, matrix_1 : np.ndarray) -> float:
    return math.sqrt(np.sum((matrix - matrix_1) ** 2))