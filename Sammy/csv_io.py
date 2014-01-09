#
# Taken from Kaggle's Getting Started With Python for Data Science
#
import numpy as np

def read_csv_np(file_path, has_header):
    with open(file_path) as f:
        if has_header: f.readline()
        data = np.array([])
        l = 0
        for line in f:
            line = line.strip().split(",")
            if l == 0:
                data = np.append(data,[float(x) for x in line])
                l = l+1
            else:
                data = np.vstack((data, [float(x) for x in line]))
    return data

def read_csv(file_path, has_header):
    with open(file_path) as f:
        if has_header: f.readline()
        data = array([])
        for line in f:
            line = line.strip().split(",")
            data.append([float(x) for x in line])
    return data

def write_csv(file_path, data):
    with open(file_path, "w") as f:
        for line in data: f.write(",".join(line) + "\n")
