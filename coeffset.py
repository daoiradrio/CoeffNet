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
        #rose_path = os.path.join(self.rose_dir, f"ROSE_C_{self.mols[idx]}.dat")
        #raw_C_rose = np.loadtxt(rose_path)
        delta_path = os.path.join(self.delta_dir, f"DELTA_C_{self.mols[idx]}.dat")
        raw_C_delta = np.loadtxt(delta_path)
        xyz_path = os.path.join(self.xyz_dir, f"{self.mols[idx]}.xyz")
        num_atoms, elems, coords = read_xyz_file(xyz_path)

        num_aos = int(raw_C_dftb.shape[0])
        C_dftb = np.zeros((num_aos, num_atoms, 1, 4, 1))
        #C_rose = np.zeros((num_aos, num_atoms, 1, 4, 1))
        C_delta = np.zeros((num_aos, num_atoms, 1, 4, 1))
        
        iao = 0
        for ielem, elem in enumerate(elems):
            if elem == "H":
                C_dftb[:, ielem, 0, 0, 0] = raw_C_dftb[iao, :]
                #C_rose[:, ielem, 0, 0, 0] = raw_C_rose[iao, :]
                C_delta[:, ielem, 0, 0, 0] = raw_C_delta[iao, :]
                iao += 1
            else:
                C_dftb[:, ielem, 0, :, 0] = raw_C_dftb[iao:iao+4, :].T
                #C_rose[:, ielem, 0, :, 0] = raw_C_rose[iao:iao+4, :].T
                C_delta[:, ielem, 0, :, 0] = raw_C_delta[iao:iao+4, :].T
                iao += 4

        return C_dftb, C_delta, elems, coords

    
'''
    def pad_collate_fn(self, batch):
        batch_C_dftb, batch_C_rose, batch_C_delta = zip(*batch)

        batch_size = len(batch_C_dftb)
        max_num_mos = max([C_dftb.shape[0] for C_dftb in batch_C_dftb])
        max_num_atoms = max([C_dftb.shape[1] for C_dftb in batch_C_dftb])

        pad_mask = np.zeros((batch_size, max_num_mos, max_num_mos, 1))

        pad_C_dftb = []
        pad_C_rose = []
        pad_C_delta = []
        for i, (C_dftb, C_rose, C_delta) in enumerate(zip(batch_C_dftb, batch_C_rose, batch_C_delta)):
            num_mos = C_dftb.shape[0]
            pad_mask[i, num_mos:max_num_mos, num_mos:max_num_mos] = -np.inf
            mo_pad_len = max_num_mos - C_dftb.shape[0]
            atom_pad_len = max_num_atoms - C_dftb.shape[1]
            pad_C_dftb.append(
                torch.nn.functional.pad(
                    torch.from_numpy(C_dftb),
                    (0, 0, 0, 0, 0, 0, 0, atom_pad_len, 0, mo_pad_len),
                    value=0
                ).numpy()
            )
            pad_C_rose.append(
                torch.nn.functional.pad(
                    torch.from_numpy(C_rose),
                    (0, 0, 0, 0, 0, 0, 0, atom_pad_len, 0, mo_pad_len),
                    value=0
                ).numpy()
            )
            pad_C_delta.append(
                torch.nn.functional.pad(
                    torch.from_numpy(C_delta),
                    (0, 0, 0, 0, 0, 0, 0, atom_pad_len, 0, mo_pad_len),
                    value=0
                ).numpy()
            )
        pad_batch_C_dftb = np.stack(pad_C_dftb)
        pad_batch_C_rose = np.stack(pad_C_rose)
        pad_batch_C_delta = np.stack(pad_C_delta)

        return pad_batch_C_dftb, pad_batch_C_rose, pad_batch_C_delta, pad_mask
'''
