import numpy as np 
from sklearn.datasets import load_iris
from common.constant import *

def iris_data_loader():
	iris_data = load_iris()

	X = iris_data.data
	label = iris_data.target

	X = poormax(X)
	X = X.T

	return X, label


def test_data_1():
	data = np.array([
		[0.00, 4.50], 
		[0.00, 5.50],
		[1.75, 4.50], 
		[1.75, 5.50],
		[3.50, 4.50], 
		[3.50, 5.50],
		[5.25, 4.50],
		[5.25, 5.50],
		[7.00, 4.50],
		[7.00, 5.50],
		[9.00, 0.00], 
		[9.00, 2.50],
		[9.00, 5.00], 
		[9.00, 7.50],
		[9.00, 10.00], 
		[10.00, 0.00],
		[10.00, 2.50], 
		[10.00, 5.00],
		[10.00, 7.50], 
		[10.00, 10.00]
	])
	data = poormax(data)
	data = data.T
	label = np.array([0] * 10 + [1] * 10)

	return data, label

def test_data(m: int, n: int) -> np.ndarray:
	"""
	制造正态分布数据，列数据 \n
	:param m: 单条数据的维度 \n
	:param n: 数据条数 \n
	"""
	x = np.ones((m, n))
	y = np.ones(n, dtype = np.int32)
	x0 = np.random.normal(10 * x, 0.3)
	x1 = np.random.normal(-3 * x, 0.5)


	x[1] = 0

	X = np.hstack((x0, x1))
	y = np.hstack((y, y * 2))

	return X, y

def poormax(X : np.ndarray, feature_axis = 1) -> np.ndarray:
	"""
		对数据进行极差化 \n
		:param feature_axis: 各特征所在的维度 \n 
		feature_axis = 1 表示每列是不同的特征 \n
	"""
	if not feature_axis:
		X = X.T
	_min = np.min(X, axis = 0)
	_max = np.max(X, axis = 0)

	across = _max - _min

	X = (X - _min) / across

	if not feature_axis:
		X = X.T
		
	return X

def __softmax__(a : np.ndarray) -> np.ndarray:
	exp = np.exp(a - np.max(a))
	return exp / np.sum(exp)

def softmax(X : np.ndarray, feature_axis = 1) -> np.ndarray:
	"""
		对数据进行softmax \n
		:param feature_axis: 各特征所在的维度 \n 
		feature_axis = 1 表示每列是不同的特征 \n
	"""
	if not feature_axis:
		X = X.T
	
	for j in range(X.shape[MatrixShapeIndex.column]):
		X[:, j] = __softmax__(X[:, j])

	if not feature_axis:
		X = X.T
	
	return X

def normalize(X : np.ndarray, feature_axis = 1) -> np.ndarray:
	"""
		对数据进行归一化 \n
		:param feature_axis: 各特征所在的维度 \n 
		feature_axis = 1 表示每列是不同的特征 \n
	"""
	if not feature_axis:
		X = X.T
	
	_sum = np.sum(X, axis = 0)

	for j in range(len(_sum)):
		if _sum[j] > 1e-100:
			X[:, j] /= _sum[j]
		else:
			X[:, j] = 1 / X.shape[MatrixShapeIndex.row]
	
	# X /= _sum
	if not feature_axis:
		X = X.T
		
	return X
	
def standardize(X : np.ndarray, feature_axis = 1) -> np.ndarray:
	"""
		对数据进行标准化 \n
		使其化为均值为 0， 标准差为 1 \n
		:param feature_axis: 各特征所在的维度 \n 
		feature_axis = 1 表示每列是不同的特征 \n
	"""
	if not feature_axis:
		X = X.T

	std = np.std(X, axis = 0)
	mean = np.mean(X, axis = 0)
	
	X = (X - mean) / std
	
	if not feature_axis:
		X = X.T
		
	return X
	
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	# x, label = iris_data_loader()
	x, label = test_data_1()
	plt.scatter(x[0], x[1], c = label, cmap = 'RdYlGn')
	plt.show()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	