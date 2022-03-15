import csv, pickle, json, os, math, sys
from collections import OrderedDict
# from bioservices import UniProt
import numpy as np
import pickle
from numpy import loadtxt

# # save numpy array as csv file
# from numpy import asarray
# from numpy import savetxt
import pickle

k_fold = 9
fname = "total_train.txt"
fpath = "createfold/kiba/"

def read_file(fname, datafolder):
    filepath = datafolder + fname
    lines = loadtxt(filepath, comments="#", delimiter=",", unpack=False)
    x = lines.tolist()
    x = list(map(int, x))
    return x
    
def create_test_set(x, datafolder):
    outpath = datafolder + "folds/"
    if not os.path.exists(outpath):
        os.makedirs(fpath + "folds/")
    outfile = outpath + "test_fold_setting.txt"
    r1 = int(0.9 * len(x))
    x1 = x[0:r1]
    print(x1)
    x2 = x[r1:]
#     x2 = x2.tolist()
#     x2 = list(map(int, x2))
    print(x2)
#     with open(outfile, 'w') as file:
#         file.write(','.join(str(int(element)) for element in x2))
    json.dump(x2, open(outfile,"w"))
    return x1, x2
    
#no of elements should be divided by fold
def create_train_folds(x, fold, datafolder):
    outpath = datafolder + "folds/"
    if not os.path.exists(outpath):
        os.makedirs(fpath + "folds/")
    outfile = outpath + "train_fold_setting.txt"
#     print(x)

    y = np.random.permutation(x)

    y = np.split(y, fold)
    for i in range(len(y)):
        y[i] = y[i].tolist()
    
    json.dump(y, open(outfile,"w"))
    return y
    
    
outer_train_sets = read_file(fname, fpath)
train_sets, test_set = create_test_set(outer_train_sets, fpath)
train_sets = create_train_folds(train_sets, k_fold, fpath)