import csv, pickle, json, os, math, sys
from collections import OrderedDict
from bioservices import UniProt
import numpy as np
import pickle


# PROT_FILE = "proteins.fasta"
PROT_FILE = "proteins.txt"
# CHEM_FILE = "ligands.txt"
# CHEM_FILE = "ligands.tab"
CHEM_FILE = "ligands_iso.txt"
# AFF_FILE = "Y.txt"
AFF_FILE = "Y_kiba.txt"
fpath = "dataprepre/kiba/"
test = True

def read_chemicals(datafolder):
    counter =0
    print(datafolder)
    print('hetre')
    filepath = datafolder + CHEM_FILE
    if  os.path.exists(filepath):
        print('found')
    else:
        print('not found')
    chemicals = {}
    with open(filepath) as file:
         print('opened')
         next(file)
         for row in file:
            chem_id = row.split('\t')[0]
            smiles = (row.split('\t')[1]).strip()
            chemicals[chem_id] = smiles
            counter +=1

    print("%d number(s) of chemical(s)" % counter)
    json.dump(chemicals, open(datafolder + 'ligands.txt', 'w'))

    return chemicals


def read_proteins(datafolder):
    proteins = {}
    counter =0
    fa=""
    filename = datafolder + PROT_FILE
    with open(filename) as f:
        fa = f.readlines()

    idindex=[]
    for i, line in enumerate(fa):
        if ">" in line:
            idindex.append(i)
    idindex.append(i)

    for i, idx in enumerate(idindex):
        print(i)
        print(idx)
        print(idindex)
        if i < len(idindex)-1:
            idx1 = idindex[i+1]
            info = fa[idx].split()

            pid = info[0][4:10]
            seq = "".join(fa[idx+1:idx1])
            seq = seq.replace("\n","")
            proteins[pid] = seq
            counter +=1

    print("%d number(s) of protein(s)" % counter)
    json.dump(proteins, open(datafolder + 'proteins.txt', 'w'))

    return proteins

prots = read_proteins(fpath)
chems = read_chemicals(fpath)
Y = np.zeros((len(chems), len(prots)))


if  os.path.exists(fpath + AFF_FILE):
    Y = np.loadtxt(fpath + AFF_FILE)

pickle.dump(Y, open(fpath + "Y","wb"), protocol=pickle.HIGHEST_PROTOCOL)
label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)

#json.dump(linepos, open(FLAGS.test_path  + "csv_pos_match.txt","w"))
indic = set(range(len(label_row_inds)))
indic = sorted(indic, key=os.urandom)


if not os.path.exists(fpath + "folds/"):
    os.makedirs(fpath + "folds/")
# if test:
json.dump(indic, open(fpath + "folds/test_fold.txt","w"))
# else:
json.dump(indic, open(fpath + "folds/train_fold.txt","w"))
