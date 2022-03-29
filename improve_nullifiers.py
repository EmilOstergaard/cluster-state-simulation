import csv
from methods import *
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

filename = 'all_modes_4dB.csv'
new_filename = 'final_improved_test4.csv'

sqz_limit = 4.0
results = []
counter_stop = 5

with open(filename) as f:
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        results.append(line)

with open(new_filename, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Squeezing [dB]','X', 'P', 'Bipartition ID', 'Bipartition 1', 'Bipartition 2']) 

def improve_sqz(result):
    value_final, x_id, p_id, bipartition_id, bipartition_info_0, bipartition_info_1 = result
    if float(value_final) > sqz_limit:
        counter = 1
        running = True
        while running:
            new_value, new_x_id, new_p_id, bipartition_id, bipartition_info_0, bipartition_info_1 = find_solution(bipartitions[int(bipartition_id)], sqz_limit=sqz_limit, op_stop=2)
            if new_value < sqz_limit or counter > counter_stop:
                running = False
            counter+=1
        if new_value < float(value_final):
            with open(new_filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([new_value, new_x_id, new_p_id, bipartition_id, bipartition_info_0, bipartition_info_1])
        else:
            with open(new_filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(result)
    else:
        with open(new_filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(result)

with Pool(cpu_count()) as pool:
        r = list(tqdm(pool.imap(improve_sqz, results), total=len(results)))
