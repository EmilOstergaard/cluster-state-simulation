import numpy as np
import math
import itertools
from multiprocessing import Pool, cpu_count
import csv
from tqdm import *

sqz_limit = 4
remaining_modes = 0

with open('data.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        if float(line[0]) > sqz_limit:
            remaining_modes += 1
        
print(remaining_modes)