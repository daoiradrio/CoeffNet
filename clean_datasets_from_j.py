import os
import numpy as np



dataset_path = "/Users/dario/datasets/C_sets/full_molecule/medium_train_set"
dftb_path = f"{dataset_path}/DFTB"
rose_path = f"{dataset_path}/ROSE"
delta_path = f"{dataset_path}/DELTA"

dftb_files = os.listdir(dftb_path)
rose_files = os.listdir(rose_path)
delta_files = os.listdir(delta_path)

for file in dftb_files:
    clean_flag = False
    file_path = f"{dftb_path}/{file}"
    check_data = np.loadtxt(file_path, dtype=str)
    n = check_data.shape[0]
    correct_data = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            correct_data[i][j] = np.real(correct_data[i][j])
            if "j" in check_data[i][j]:
                clean_flag = True
    if clean_flag:
        np.savetxt(file_path, correct_data)

for file in rose_files:
    clean_flag = False
    file_path = f"{rose_path}/{file}"
    check_data = np.loadtxt(file_path, dtype=str)
    n = check_data.shape[0]
    correct_data = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            correct_data[i][j] = np.real(correct_data[i][j])
            if "j" in check_data[i][j]:
                clean_flag = True
    if clean_flag:
        np.savetxt(file_path, correct_data)

for file in delta_files:
    clean_flag = False
    file_path = f"{delta_path}/{file}"
    check_data = np.loadtxt(file_path, dtype=str)
    n = check_data.shape[0]
    correct_data = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            correct_data[i][j] = np.real(correct_data[i][j])
            if "j" in check_data[i][j]:
                clean_flag = True
    if clean_flag:
        np.savetxt(file_path, correct_data)
