import numpy as np 
import pandas as pd
from sklearn import metrics

def get_cluster(U : np.ndarray) -> np.ndarray:
    return np.argmax(U, axis = 0)

def NMI(y : np.ndarray, label : np.ndarray):
    return metrics.normalized_mutual_info_score(label, y)

def AMI(y : np.ndarray, label : np.ndarray):
    return metrics.adjusted_mutual_info_score(label, y)

def ARI(y : np.ndarray, label : np.ndarray):
    return metrics.adjusted_rand_score(label, y)

def ACC(y : np.ndarray, label : np.ndarray):
    return metrics.accuracy_score(label, y)

f_list = [NMI, AMI, ARI, ACC]
indexs_name_list = ['NMI', 'AMI', 'ARI', 'ACC']

def get_all_indices(y : np.ndarray, label : np.ndarray) -> np.ndarray:
    indices = np.zeros(4)
    
    for i in range(4):
        indices[i] = f_list[i](y, label)
    
    return pd.Series(data = indices, index = indexs_name_list)

if __name__ == '__main__':
    a = [1] * 20
    b = [0] * 10 + [1] * 10
    print(a)
    print(b)

    print(NMI(a, b))
    print(AMI(a, b))