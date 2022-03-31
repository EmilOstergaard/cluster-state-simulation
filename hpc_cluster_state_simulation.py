from ast import operator
import numpy as np
import math
import itertools
from multiprocessing import Pool, cpu_count
import csv
from tqdm import *
import pickle

class Nullifier:
    def __init__(self, unit_cell_modes = np.zeros(16), boundary_modes = np.zeros(112)):
        self.unit_cell_modes = unit_cell_modes
        self.boundary_modes = boundary_modes

    def __add__(self, n):
        return Nullifier(self.unit_cell_modes + n.unit_cell_modes, self.boundary_modes + n.boundary_modes)
    
    def __sub__(self, n):
        return Nullifier(self.unit_cell_modes - n.unit_cell_modes, self.boundary_modes - n.boundary_modes)

with open('operators.csv', 'r') as operators_file:
    csvreader = csv.reader(operators_file)
    possible_nullifier_combinations = pickle.load(operators_file)

with open('bipartitions.obj', 'rb') as bipartitions_file:
    bipartitions = pickle.load(bipartitions_file)
    

def find_solution(bipartition):
    sqz_limit = 4
    value = np.infty
    value_raw = 0
    bipartition_matrix = bipartition[0]
    bipartition_id = bipartition[1]
    bipartition_info = bipartition[2]

    for [x_operator, p_operator, num_operators, x_operator_iden, p_operator_iden] in possible_nullifier_combinations:
        h_j1 = bipartition_matrix[0] * x_operator.unit_cell_modes
        g_j1 = bipartition_matrix[0] * p_operator.unit_cell_modes
        h_j2 = bipartition_matrix[1] * x_operator.unit_cell_modes
        g_j2 = bipartition_matrix[1] * p_operator.unit_cell_modes
        try:
            temp_value = abs(h_j1 @ g_j1)+abs(h_j2 @ g_j2)
            if temp_value > value_raw:
                value_raw = temp_value
                value = abs(10*math.log10(value_raw/(8 * num_operators)))
                operators = x_operator_iden, p_operator_iden
        except:
            pass

        if value <= sqz_limit:
            break

    with open('cluster_state_simulation_data1.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow([value, operators, bipartition_id, bipartition_info[0], bipartition_info[1]])

if __name__ == '__main__':

    print('Starting search...')
    bipartitions = bipartitions[1]

    with open('cluster_state_simulation_data1.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(['Squeezing [dB]','Operators','Bipartition ID', 'Bipartition 1', 'Bipartition 2'])
    
    print('Running on %d cores'%cpu_count())
    with Pool(cpu_count()) as pool:
        r = list(tqdm(pool.imap(find_solution, bipartitions), total=len(bipartitions)))