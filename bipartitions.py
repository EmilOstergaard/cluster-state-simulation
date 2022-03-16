import numpy as np
import math
import pickle

def binary(x,n):
    num = x
    bin_num = np.zeros(n)

    for i in range(n-1,-1,-1):
        if num - 2**i >= 0:
            bin_num[i] = int(1)
            num = num - 2**i

    return [int(x) for x in bin_num.tolist()]

def binary_inv(x,n):
    num = x
    bin_num = np.ones(n)

    for i in range(n-1,-1,-1):
        if num - 2**i >= 0:
            bin_num[i] = int(0)
            num = num - 2**i

    return [int(x) for x in bin_num.tolist()]

num_modes = 16
bipartitions = []

for i in range(2**(num_modes-1)-1):
    bin_num = binary(i+1, num_modes)
    bin_num_inv = binary_inv(i+1, num_modes)
    partition_1 = []
    partition_2 = []
    for k in range(num_modes):
        if bin_num[k] == 1:
            partition_1.append(k+1)
        else:
            partition_2.append(k+1)
    
    partition_1_matrix = np.asarray(bin_num, dtype='float')
    partition_2_matrix = np.asarray(bin_num_inv, dtype='float')

    bipartition_id = i
    bipartitions.append([[partition_1_matrix, partition_2_matrix], bipartition_id, (partition_1, partition_2)])

with open('bipartitions.obj', 'wb') as f:
    pickle.dump(bipartitions, f)