import os
import sys

import numpy as np
import scipy.sparse.csgraph
import pprint
from sklearn import preprocessing as p

size_of_grid = 3
threshold = 0.1

# import csv to np array
from numpy import loadtxt
data = np.loadtxt("Education_Indicators_2014_norm.csv", delimiter=';', skiprows=1, usecols=range(1,10))
# get first column data index
data_index = np.loadtxt("Education_Indicators_2014_norm.csv", delimiter=';', skiprows=1, usecols=range(0,1), dtype=str)
# get first row data name
data_name = np.loadtxt("Education_Indicators_2014_norm.csv", delimiter=';', skiprows=0, usecols=range(1,10), dtype=str, max_rows=1)

# #  normalize data
number_of_features = data.shape[1]
# # remove the first column

# normalize data with preprocessing
data = p.normalize(data, norm='l2', axis=0, copy=True, return_norm=False)

number_of_features = data.shape[1]
number_of_samples = data.shape[0]

grid = np.zeros((size_of_grid, number_of_features))
for f in range(number_of_features):
    for e in data[:, f]:
        grid[int(e * size_of_grid), f] += 1
pprint.pprint(grid)
is_dense = grid > threshold * number_of_samples

# #  find dense units
dense_units = []
for f in range(number_of_features):
    for g in range(size_of_grid):
        if is_dense[g, f]:
            dense_unit = dict({f: g})
pprint.pprint(dense_units)




# function to find 1 dimensional dense units



