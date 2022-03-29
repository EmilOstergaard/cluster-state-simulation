from new_approach import *
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

filename = 'final_data_all_modes.csv'

bipartitions = bipartitions[:1000]

def save_solution(bipartition):
    with open(filename, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(find_solution(bipartition, 2))

if __name__ == '__main__':

    with open(filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(['Squeezing [dB]','X', 'P', 'Bipartition ID', 'Bipartition 1', 'Bipartition 2'])

    with Pool(cpu_count()-2) as pool:
        r = list(tqdm(pool.imap(save_solution, bipartitions), total=len(bipartitions)))
    