from fileinput import close
from os import remove
from time import time
import numpy as np
import math
import itertools
from multiprocessing import Pool, cpu_count
import csv
import pickle
from tqdm import *

print('test')

with open('bipartitions.obj', 'rb') as f:
    bipartitions = pickle.load(f)
    close

print('Loaded bipartitions')

def find_solution(bipartition):
    sqz_limit = 4
    bipartition_matrix = bipartition[0]
    bipartition_id = bipartition[1]
    bipartition_info = bipartition[2]

    h_j1 = bipartition_matrix[0] * x_operator_unit
    g_j1 = bipartition_matrix[0] * p_operator_unit
    h_j2 = bipartition_matrix[1] * x_operator_unit
    g_j2 = bipartition_matrix[1] * p_operator_unit
    try:
        value = abs(10*math.log10(abs(h_j1 @ g_j1) + abs(h_j2 @ g_j2)/(8*num_operators)))
        operators_iden = [x_operator_iden, p_operator_iden]
        if value <= sqz_limit:
            # remaining_partitions.pop(bipartition_id)
            remove_bipartitions.append(bipartition_id)
            print(len(remove_bipartitions))
            with open('final_data_all_modes.csv', 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow([value, operators_iden[0], operators_iden[1], bipartition_id, bipartition_info[0], bipartition_info[1]])
    except:
        pass
            
    

if __name__ == '__main__':

    print('Starting search...')
    bipartitions = bipartitions[:1000]
    global remove_bipartitions

    with open('final_data_all_modes.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(['Squeezing [dB]','X', 'P', 'Bipartition ID', 'Bipartition 1', 'Bipartition 2'])

    with open('operators.csv', 'r') as f:
        csvreader = csv.reader(f)
        remaining_partitions = bipartitions
        print('test2')
        for [x_operator_unit, _, p_operator_unit, _, num_operators, x_operator_iden, p_operator_iden] in csvreader:
            x_operator_unit = np.array(x_operator_unit[1:-1].split(), dtype = 'float')
            p_operator_unit = np.array(p_operator_unit[1:-1].split(), dtype = 'float')
            num_operators = int(num_operators)

            remove_bipartitions = []
            with Pool(cpu_count()) as pool:
                r = list(tqdm(pool.imap(find_solution, remaining_partitions), total=len(remaining_partitions)))
            
            print('removing solved bipartitions')
            print(len(remove_bipartitions))
            for i in remove_bipartitions:
                remaining_partitions.pop(i)
            print(len(remaining_partitions))