from asyncio.base_subprocess import ReadSubprocessPipeProto
import csv
from wsgiref.handlers import read_environ
from matplotlib import pyplot

sqz_limit = 3
remaining_modes = 0
highest = 0
total = 0
values = []

with open('all_modes_36db.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        total += 1
        values.append(float(line[0]))
        if float(line[0]) > sqz_limit:
            remaining_modes += 1
        if float(line[0]) > highest:
            highest = float(line[0])
        
print(remaining_modes)
print(total)
print(highest)

pyplot.hist(values, bins=100)
pyplot.show()