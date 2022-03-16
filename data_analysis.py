import csv

sqz_limit = 3.1
remaining_modes = 0
highest = 0
total = 0

with open('final_data_4dB_1000.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        total += 1
        if float(line[0]) > sqz_limit:
            remaining_modes += 1
        if float(line[0]) > highest:
            highest = float(line[0])
        
print(remaining_modes)
print(total)
print(highest)