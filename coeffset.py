import os
import e3x
import torch

import numpy as np

from torch.utils.data import Dataset
from CompChemUtils.files import read_xyz_file
from jax import numpy as jnp
    


class CoeffSet(Dataset):

    def __init__(self, annotation_file, dftb_dir, delta_dir, xyz_dir):
        self.mol_idx = np.loadtxt(annotation_file, usecols=2, dtype=str)
        self.C_idx = np.loadtxt(annotation_file, usecols=3, dtype=int)
        self.dftb_dir = dftb_dir
        self.delta_dir = delta_dir
        self.xyz_dir = xyz_dir


    def __len__(self):
        return self.mol_idx.size


    def __getitem__(self, i):
        dftb_coeff_vec = np.loadtxt(os.path.join(self.dftb_dir, f"DFTB_C_{self.mol_idx[i]}_{self.C_idx[i]}.dat"))
        delta_coeff_vec = np.loadtxt(os.path.join(self.delta_dir, f"DELTA_C_{self.mol_idx[i]}_{self.C_idx[i]}.dat"))
        natoms, elems, coords = read_xyz_file(os.path.join(self.xyz_dir, f"{self.mol_idx[i]}.xyz"))

        x_dftb = np.zeros((natoms, 2, 4, 1))
        y_delta = np.zeros((natoms, 2, 4, 1))
        iao = 0
        for i, elem in enumerate(elems):
            if elem == "H":
                x_dftb[i, 0, 0, 0] = dftb_coeff_vec[iao]
                y_delta[i, 0, 0, 0] = delta_coeff_vec[iao]
                iao += 1
            else:
                x_dftb[i, 1, :, 0] = dftb_coeff_vec[iao : iao+4]
                y_delta[i, 1, :, 0] = delta_coeff_vec[iao : iao+4]
                iao += 4

        coords = jnp.asarray(coords)

        return x_dftb, y_delta, coords
