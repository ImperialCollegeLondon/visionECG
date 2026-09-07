from torch.utils.data import Dataset
import torch
import numpy as np
import csv
import pandas as pd
import h5py


class MeshVAEDataset(Dataset):
    """Load MeshVAE vertices, faces, and edges."""

    def __init__(self, config, data_usage='train'):
        data_list = []
        csvpath = f"{config.label_dir}/mesh_{data_usage}.csv"

        try:
            df = pd.read_csv(csvpath)
            for _, row in df.iterrows():
                data_list.append(row['Unnamed: 0'])
        except Exception as e:
            print(f"Error reading CSV {csvpath}: {e}")
            with open(csvpath, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data_list.append(row['Unnamed: 0'])

        self.data_list = data_list
        self.data_dir = config.target_seg_dir
        self.device = config.device
        self.seq_len = config.seq_len
        self.label_dir = config.label_dir
        self.normalize = config.normalize
        self.n_samples = config.n_samples
        self.surf_type = config.surf_type

    def __getitem__(self, index):
        subid = int(self.data_list[index])
        mesh_path = f"{self.data_dir}/{subid}/image_space_pipemesh"

        if self.surf_type == 'all':
            h5filepath = f'{mesh_path}/preprossed_vtk.hdf5'
        elif self.surf_type == 'sample':
            h5filepath = f'{mesh_path}/preprossed_decimate.hdf5'

        f = h5py.File(h5filepath, "r")
        mesh_verts = torch.Tensor(np.array(f['heart_v']))
        mesh_faces = torch.LongTensor(np.array(f['heart_f']))
        mesh_edges = torch.LongTensor(np.array(f['heart_e']))

        return (mesh_verts, mesh_faces, mesh_edges, subid)

    def __len__(self):
        return len(self.data_list)

    @property
    def patient_ids(self):
        return self.data_list
