对于所有算法，参数设置如下

|    参数     |              解释               |
| :---------: | :-----------------------------: |
|      X      |  样本数据集，每个样本为列向量   |
| cluster_num |             类簇数              |
|   [gamma]   |     惩罚项系数，默认为0.01      |
|     [m]     | 模糊聚类算法的模糊度值，默认为2 |

#### 半监督信息

半监督信息通过类中 inputSemi_SupervisedInformaintion 方法输入。参数设置如下

|   参数    |            解释            |
| :-------: | :------------------------: |
|  tilde_U  | 先验隶属度矩阵 $\tilde{U}$ |
| initial_V |      初始聚类中心矩阵      |

如果不输入半监督信息，则默认先验隶属度矩阵为全0矩阵。初始聚类中心为在样本取值空间中随机生成。

#### 执行聚类

聚类过程通过类中的 iteration 方法实现。参数设置如下

|      参数      |                  解释                  |
| :------------: | :------------------------------------: |
| [quit_epsilon] | 迭代终止条件 $\varepsilon$，默认为1e-5 |
|   [iter_num]   |      算法最大迭代次数，默认为200       |

iteration 返回的是一个字典，说明如下表

| 键值 |                解释                |
| :--: | :--------------------------------: |
|  U   | 隶属度矩阵，聚类结果需根据它来得出 |
|  V   |            聚类中心矩阵            |
|  t   |              迭代次数              |
| U_f  |     最后两个隶属度矩阵的Frobenius范数      |
| V_f  |     最后两个聚类中心矩阵的Frobenius范数      |
| use_time  |     算法运行用时，单位为s      |
使用示例：

```python
''' 数据准备 '''
from sklearn.datasets import load_iris
X, label = load_iris()	# 使用莺尾花数据集
X = Deviation(X)		# 对数据进行离差标准化
X = X.T
c = 3
```

```python
from algorithms import MEC

MECa = MEC.MECA(X, c)
clusterResult = MECa.iteration()	
U = clusterResult['U']
```

