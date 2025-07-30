import numpy as np
import os
import torch


coords_al4=[]
for i in range(5):
    file_name='../generated/al4_{}_numpy_dict.npz'.format(i)
    data = np.load(file_name, allow_pickle=True)
    coords_al4.append(data['samples'])
coords_al4 = np.concatenate(coords_al4, axis=0)
np.save('../data/AAAA/train_gen_coords.npy', coords_al4)

coords_al4=[]
for i in range(5,10):
    file_name='../generated/al4_{}_numpy_dict.npz'.format(i)
    data = np.load(file_name, allow_pickle=True)
    coords_al4.append(data['samples'])
coords_al4 = np.concatenate(coords_al4, axis=0)
np.save('../data/AAAA/test_gen_coords.npy', coords_al4)


coords_al6=[]
for i in range(5):
    file_name='../generated/al6_{}_numpy_dict.npz'.format(i)
    data = np.load(file_name, allow_pickle=True)
    coords_al6.append(data['samples'])
coords_al6 = np.concatenate(coords_al6, axis=0)
np.save('../data/AAAAAA/train_gen_coords.npy', coords_al6)

coords_al6=[]
for i in range(5,10):
    file_name='../generated/al6_{}_numpy_dict.npz'.format(i)
    data = np.load(file_name, allow_pickle=True)
    coords_al6.append(data['samples'])
coords_al6 = np.concatenate(coords_al6, axis=0)
np.save('../data/AAAAAA/test_gen_coords.npy', coords_al6)