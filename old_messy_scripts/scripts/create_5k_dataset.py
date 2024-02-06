import os
import torch
from torch.utils.data import Dataset
from ast import literal_eval
import pandas as pd
import numpy as np


torch.cuda.set_device(1)


class SmallDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file).sample(5000)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding_path = os.path.join(self.root_dir, self.labels.iloc[idx,0])
        embedding = torch.load(embedding_path)[0,0,1:,:]
        label = literal_eval(self.labels.iloc[idx, 1])
        if self.transform:
            embedding = self.transform(embedding)
        return embedding, label, embedding_path

dataset_orig = SmallDataset(csv_file='../data/rmsd_dataset.csv',
                                    root_dir='/mnt/nasdata/vera/msa_transformer_embeddings/')


all_features = []
all_labels = []
all_pdb_ids = []
all_residues = []

for i in range(len(dataset_orig)):
    feature, label, id = dataset_orig[i]
    for residue in range(feature.shape[0]):
        all_features.append(feature[residue].cpu().numpy())
        all_labels.append(label[residue])
        all_pdb_ids.append(id)
        all_residues.append(residue)
    print(i)

print(len(all_features), len(all_labels))

feature_tensor = torch.from_numpy(np.stack(all_features))
torch.save(feature_tensor, '../data/5k_protein_dataset/5k_protein_tensor.pt')
label_tuples = list(zip(all_pdb_ids,all_residues,all_labels))
labels_df = pd.DataFrame(label_tuples, columns=['pdb','residue','rmsd'])
labels_df.to_csv('../data/5k_protein_dataset/5k_protein_labels.csv')

