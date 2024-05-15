import os

import numpy as np

from torch.utils.data import Dataset
from CompChemUtils.files import read_xyz_file



class CoeffSet(Dataset):

    def __init__(self, annotation_file, dftb_dir, rose_dir, delta_dir, xyz_dir):
        self.mols = np.loadtxt(annotation_file, usecols=1, dtype=str)
        self.dftb_dir = dftb_dir
        self.rose_dir = rose_dir
        self.delta_dir = delta_dir
        self.xyz_dir = xyz_dir


    def __len__(self):
        return self.mols.size
    

    def __getitem__(self, idx):
        dftb_path = os.path.join(self.dftb_dir, f"DFTB_C_{self.mols[idx]}.dat")
        raw_C_dftb = np.loadtxt(dftb_path)
        rose_path = os.path.join(self.rose_dir, f"ROSE_C_{self.mols[idx]}.dat")
        raw_C_rose = np.loadtxt(rose_path)
        delta_path = os.path.join(self.delta_dir, f"DELTA_C_{self.mols[idx]}.dat")
        raw_C_delta = np.loadtxt(delta_path)
        xyz_path = os.path.join(self.xyz_dir, f"{self.mols[idx]}.xyz")
        num_atoms, elems, coords = read_xyz_file(xyz_path)

        num_aos = int(raw_C_dftb.shape[0])
        C_dftb = np.zeros((num_aos, num_atoms, 1, 4, 1))
        C_rose = np.zeros((num_aos, num_atoms, 1, 4, 1))
        C_delta = np.zeros((num_aos, num_atoms, 1, 4, 1))
        
        iao = 0
        for ielem, elem in enumerate(elems):
            if elem == "H":
                C_dftb[:, ielem, 0, 0, 0] = raw_C_dftb[iao, :]
                C_rose[:, ielem, 0, 0, 0] = raw_C_rose[iao, :]
                C_delta[:, ielem, 0, 0, 0] = raw_C_delta[iao, :]
                iao += 1
            else:
                C_dftb[:, ielem, 0, :, 0] = raw_C_dftb[iao:iao+4, :].T
                C_rose[:, ielem, 0, :, 0] = raw_C_rose[iao:iao+4, :].T
                C_delta[:, ielem, 0, :, 0] = raw_C_delta[iao:iao+4, :].T
                iao += 4

        #return C_dftb, C_rose, C_delta, elems, coords
        return raw_C_dftb
        


train_annotation_file = "/Users/dario/datasets/C_sets/full_molecule/medium_train_set/molslist.dat"
train_dftb_dir = "/Users/dario/datasets/C_sets/full_molecule/medium_train_set/DFTB"
train_rose_dir = "/Users/dario/datasets/C_sets/full_molecule/medium_train_set/ROSE"
train_delta_dir = "/Users/dario/datasets/C_sets/full_molecule/medium_train_set/DELTA"
train_xyz_dir = "/Users/dario/preprocessed_QM9/y_medium_train_set"
valid_annotation_file = "/Users/dario/datasets/C_sets/full_molecule/medium_valid_set/molslist.dat"
valid_dftb_dir = "/Users/dario/datasets/C_sets/full_molecule/medium_valid_set/DFTB"
valid_rose_dir = "/Users/dario/datasets/C_sets/full_molecule/medium_valid_set/ROSE"
valid_delta_dir = "/Users/dario/datasets/C_sets/full_molecule/medium_valid_set/DELTA"
valid_xyz_dir = "/Users/dario/preprocessed_QM9/y_medium_valid_set"

train_dataset = CoeffSet(
    annotation_file=train_annotation_file,
    dftb_dir=train_dftb_dir,
    rose_dir=train_rose_dir,
    delta_dir=train_delta_dir,
    xyz_dir=train_xyz_dir
)

n = train_dataset.__len__()
norms = []
for i in range(n):
    C_dftb = train_dataset.__getitem__(i)
    norms.append(np.sum(np.linalg.norm(C_dftb, axis=0)))

print(np.mean(norms))
print(np.std(norms))
