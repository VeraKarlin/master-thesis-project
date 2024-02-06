#! /home/lucas/miniconda3/envs fresh

import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
_, alphabet = torch.hub.load("facebookresearch/esm:main", "esm_msa1b_t12_100M_UR50S")
from einops import rearrange

class fully_connected_net(nn.Module):
    """
    Current network I'm playing with May 10th.
    Input shape:  (N,C,L)
    N - size of batch or number of proteins
    C - number of features or in convolution terms: number of channels
    L - sequence length
    """
    def __init__(self, input_features= 768, input_length=511):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_proj = nn.Sequential(
            nn.Linear(input_features * input_length, 30),
            nn.ReLU(),
            nn.Linear(30,1)
        )

    def forward(self, msa_query_embeddings):
        msa_query_embeddings = torch.permute(msa_query_embeddings, (0, 2, 1))
        x = self.flatten(msa_query_embeddings)
        logits = self.linear_proj(x)
        return logits

class conv_net(nn.Module):
    """
    Current network I'm playing with May 11th.
    Input shape:  (N,C,L)
    N - size of batch or number of proteins
    C - number of features or in convolution terms: number of channels
    L - sequence length
    """
    def __init__(self, input_features= 768, input_length=511):
        super().__init__()
        self.linear_proj = nn.Sequential(
            torch.nn.BatchNorm1d(num_features=input_features),
            nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=5, groups=input_features),
            nn.AvgPool1d(kernel_size=3),
            nn.Conv1d(in_channels=input_features, out_channels=input_features//3, kernel_size=1),
            nn.AvgPool1d(kernel_size=5),
            nn.Conv1d(in_channels=input_features//3, out_channels=10, kernel_size=1),
            nn.Flatten(),
            nn.Linear(in_features=10*33,out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, msa_query_embeddings):
        msa_query_embeddings = torch.permute(msa_query_embeddings, (0, 2, 1)) #from (N,L,C) to (N,C,L)  n is number of
        pred = self.linear_proj(msa_query_embeddings)
        return pred
    
class conv_attent_net(nn.Module):
    """
    Convolutional network that takes an attention map as input
    Input shape:  (N,L,L)
    N - size of batch or number of proteins
    L - sequence length
    """
    def __init__(self, input_length=511):
        super().__init__()
        self.convolutions = nn.Sequential(
            #[n, 12, 511, 511]
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=7, groups=1),
            #[n, 20, 505, 505]
            nn.AvgPool2d(kernel_size=3, stride=3),
            #[n, 10, 168, 168]
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3),
            #[n, 20, 166, 166]
            nn.AvgPool2d(kernel_size=2, stride=2),
            #[n, 20, 81, 81]
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=10),
            #[n, 40, 72, 72]
            nn.AvgPool2d(kernel_size=2, stride=2)
            #[n, 40, 37, 37]
        )

        self.linear_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=40*37*37,out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1)
            # nn.Linear(in_features=100, out_features=10),
            # nn.ReLU(),
            # nn.Linear(in_features=10,out_features=1)
        )

    def forward(self, map):
        batch_size = map.shape[0]
        map_length = map.shape[1]
        conv_map = torch.reshape(map, (batch_size, 1, map_length, map_length))
        conv_map = self.convolutions(conv_map)
        pred = self.linear_proj(conv_map)    
        return pred
    
class conv_multi_attent_net(nn.Module):
    """
    Convolutional network that takes an attention map as input
    Input shape:  (N,L,L)
    N - size of batch or number of proteins
    L - sequence length
    """
    def __init__(self, input_features = 12, input_length=511):
        super().__init__()
        self.convolutions = nn.Sequential(
            #[n, 12, 511, 511]
            nn.Conv2d(in_channels=input_features, out_channels=input_features*2, kernel_size=7, groups=input_features),
            #[n, 24, 505, 505]
            nn.AvgPool2d(kernel_size=3, stride=3),
            #[n, 24, 168, 168]
            nn.Conv2d(in_channels=input_features*2, out_channels=input_features*4, kernel_size=3),
            #[n, 48, 166, 166]
            nn.AvgPool2d(kernel_size=2, stride=2),
            #[n, 48, 81, 81]
            nn.Conv2d(in_channels=input_features*4, out_channels=40, kernel_size=10),
            #[n, 40, 72, 72]
            nn.AvgPool2d(kernel_size=2, stride=2)
            #[n, 40, 37, 37]
        )

        self.linear_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=40*37*37,out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1)
            # nn.Linear(in_features=100, out_features=10),
            # nn.ReLU(),
            # nn.Linear(in_features=10,out_features=1)
        )

    def forward(self, map):
        batch_size = map.shape[0]
        map_length = map.shape[-1]
        #conv_map = torch.reshape(map, (batch_size, 12, map_length, map_length))
        conv_map = self.convolutions(map)
        pred = self.linear_proj(conv_map)    
        return pred

class S_pred_rebuild(nn.Module):
    """
    Convolutional network that takes attentiona and embeddings as input
    Input shape:  (N,L,L)
    N - size of batch or number of proteins
    L - sequence length
    """
    def __init__(self):
        super().__init__()
        self.MLP1 = nn.Sequential(
            nn.Linear(in_features=768, out_features=384),
            nn.LayerNorm(normalized_shape=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=192),
            nn.LayerNorm(normalized_shape=192),
            nn.ReLU(),
            nn.Linear(in_features=192, out_features=192)
        )

        self.Normalization = nn.LayerNorm(normalized_shape=768)

        self.LSTM = nn.LSTM(
            input_size=336,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

        self.MLP2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=124),
            nn.LayerNorm(normalized_shape=(124)),
            nn.ReLU(),
            nn.Linear(in_features=124, out_features=64),
            nn.LayerNorm(normalized_shape=(64)),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

        self.MLP3 = nn.Sequential(
            nn.Linear(in_features=511, out_features=224),
            nn.LayerNorm(normalized_shape=(224)),
            nn.ReLU(),
            nn.Linear(in_features=224, out_features=64),
            nn.LayerNorm(normalized_shape=(64)),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, embeddings, attentions):
        #embeddings = torch.permute(embeddings, (0, 2, 1)) #from (N,L,F) to (N,F,L)  n is batch size, C, number of features, L sequence length.        
        emb = self.MLP1(embeddings)

        attentions = torch.permute(attentions, (0, 3, 4, 1, 2)) #from (12, 12, L, L,) to (L, L, 12 ,12 ) 
        attentions = torch.reshape(attentions, [attentions.shape[0], 511, 511, 144]) #collate (L, L, 12, 12) to (L, L, 144)
        avg_col = torch.mean(input=attentions, dim=1, keepdim=False)
        #avg_row = torch.mean(input=attentions, dim=2, keepdim=False)
        #attentions = torch.concat([avg_col, avg_row], -1)
        #attentions = attentions[:,0,:,:]

        X = torch.concat([emb, avg_col], -1)
        #X = X[:,0,:,:] #remove empty dimension so that it becomes 3D [N, L, H] = [batch size, seq length, num features] 
        Y, (_, _) = self.LSTM(X)
        #Y = torch.reshape(Y, [Y.shape[0], 1, 511, 256])

        dg = self.MLP2(Y)
        dg = dg[:,:,0]
        dG = self.avgpool(dg)
        return dG

class S_pred_copy(nn.Module):
    def __init__(self, input_feature_size=768, input_sequence_length = 511, hidden_node=256, dropout=0.25, need_row_attention=True, class_num=1):
        super().__init__()
        self.need_row_attention = need_row_attention
        self.linear_proj = nn.Sequential(
            nn.Linear(input_feature_size, input_feature_size // 2),
            nn.InstanceNorm1d(input_feature_size // 2),
            nn.ReLU(),
            nn.Linear(input_feature_size // 2, input_feature_size // 4),
            nn.InstanceNorm1d(input_feature_size // 4),
            nn.ReLU(),
            nn.Linear(input_feature_size // 4, input_feature_size // 4),
        )

        if self.need_row_attention:
            lstm_input_feature_size = input_feature_size // 4 + 144*2
        else:
            lstm1_input_feature_size = input_feature_size // 4

        self.lstm = nn.LSTM(
            input_size=lstm_input_feature_size,
            batch_first=True,
            hidden_size=hidden_node,
            num_layers=3,
            bidirectional=True,
            dropout=dropout,
        )

        self.to_property = nn.Sequential(
            nn.Linear(hidden_node * 2, hidden_node * 2),
            nn.InstanceNorm1d(hidden_node * 2),
            nn.ReLU(),
            nn.Linear(hidden_node * 2, class_num),
        )


        self.to_residue = nn.Sequential(
            nn.Linear(hidden_node * 2, hidden_node * 2),
            nn.InstanceNorm1d(hidden_node * 2),
            nn.ReLU(),
            nn.Linear(hidden_node * 2, hidden_node),
            nn.InstanceNorm1d(hidden_node),
            nn.ReLU(),
            nn.Linear(hidden_node, class_num),
        )

        self.conv_to_stability = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=20), # L = 511 - 20 + 1 = 492
            nn.BatchNorm1d(num_features=10),
            nn.Flatten(),
            nn.Linear(10 * 492, 512), #C * L = 10 * 492
            nn.InstanceNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.to_stability = nn.Sequential(
            nn.Linear(input_sequence_length, input_sequence_length // 2),
            nn.InstanceNorm1d(input_sequence_length // 2),
            nn.ReLU(),
            nn.Linear(input_sequence_length // 2, 1),
        )

    def forward(self, msa_query_embeddings, msa_row_attentions):
        msa_query_embeddings = self.linear_proj(msa_query_embeddings)

        if self.need_row_attention:
            msa_row_attentions = rearrange(msa_row_attentions, 'b l h i j -> b (l h) i j')
            msa_attention_features = torch.cat((torch.mean(msa_row_attentions, dim=2), torch.mean(msa_row_attentions, dim=3)), dim=1)
            # msa_attention_features = (torch.mean(msa_row_attentions, dim=2) + torch.mean(msa_row_attentions, dim=3))/2
            msa_attention_features = msa_attention_features.permute((0, 2, 1))

            lstm_input = torch.cat([msa_query_embeddings, msa_attention_features], dim=2)

        else:
            lstm_input = msa_query_embeddings

        lstm_output, _ = self.lstm(lstm_input)
        label_output = self.to_residue(lstm_output)
        #label_output = label_output[:,:,0]
        #output = self.to_stability(label_output)

        label_output = label_output.permute((0, 2, 1))
        output = self.conv_to_stability(label_output)


        return output

class S_pred_Dataset(Dataset):
    """ Keeps track of the the attentions and the normalized dG values."""
    def __init__(self, 
                    selection,
                    device, 
                    database_file = '/home/lucas/esmmsa/data/stability/ProTherm2.csv',
                    embed_dir = '/mnt/nasdata/lucas/data/query_embed/', 
                    attent_dir = '/mnt/nasdata/lucas/data/row_attent/',
                    only_wt = False, 
                    transform=None, 
                    target_transform=None):
        
        database = pd.read_csv(database_file)
        self.device = device
        if only_wt:
            database = database[database['MUTATION'] == 'WT']
        database['keep'] = False
        for id in selection:
            match = database['UniProt_ID'] == id
            database['keep'] += match 
        self.database = database[database['keep'] == True]
        self.embed_dir = embed_dir
        self.attent_dir = attent_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.database.shape[0]

    def __getitem__(self, idx):
        UniProtID = self.database['UniProt_ID'].iloc[idx]
        typ = self.database['MUTATION'].iloc[idx]
        name = UniProtID + '_' + typ 

        attent_path = self.attent_dir + name + '_.pt'
        attentions = torch.load(attent_path, map_location=self.device)
        attentions = padd_multi_attent(attentions, side='after')

        embed_path = self.embed_dir + name + '_.pt'
        embedding = torch.load(embed_path, map_location=self.device)
        embedding = padd_embed(embedding, side='after')

        label = self.database['dG_minmax_scale'].iloc[idx]
        #label = self.labels['dG'].iloc[idx]
        label = torch.tensor(label)

        if self.transform:
            attentions = self.transform(attentions)
        if self.target_transform:
            label = self.target_transform(label)
        return embedding, attentions, label, name

class first_net(nn.Module):
    """
    An oversimplified network.
    After testing: it cannot learn anything.
    """
    def __init__(self, input_feature_size= 768):
        super().__init__()
        self.linear_proj = nn.Sequential(
            nn.Conv1d(in_channels=511, out_channels=1, kernel_size=4),
            nn.Linear(input_feature_size - 1*3 , input_feature_size // 2),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(input_feature_size // 2, input_feature_size // 8),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(input_feature_size // 8, 1),
        )

    def forward(self, msa_query_embeddings):
        pred = self.linear_proj(msa_query_embeddings)
        return pred

class CustomDataset(Dataset):
    """ Keeps track of the the embeddings and the normalized dG values."""
    def __init__(self, 
                 labels_file,
                 selection,
                 device, 
                 embed_dir = '/mnt/nasdata/lucas/data/full_embed/', 
                 transform=None, 
                 target_transform=None):
        
        temp = pd.read_csv(labels_file)
        self.device = device
        self.labels = temp.iloc[selection]
        self.embed_dir = embed_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        name = self.labels['filename'].iloc[idx]
        embed_path = self.embed_dir + name + '.pt'
        embedding = torch.load(embed_path, map_location=self.device)
        embedding = padd_embed(embedding)
        label = self.labels['norm_dG'].iloc[idx]
        #label = self.labels['dG'].iloc[idx]
        label = torch.tensor(label)
        if self.transform:
            embedding = self.transform(embedding)
        if self.target_transform:
            label = self.target_transform(label)
        return embedding, label, name
    
class AttentionDataset(Dataset):
    """ Keeps track of the the attentions and the normalized dG values."""
    def __init__(self, 
                    labels_file,
                    selection,
                    device, 
                    attent_dir = '/mnt/nasdata/lucas/data/row_attent/', 
                    transform=None, 
                    target_transform=None):
        
        temp = pd.read_csv(labels_file)
        self.device = device
        self.labels = temp.iloc[selection]
        self.attent_dir = attent_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        name = self.labels['filename'].iloc[idx]
        attent_path = self.attent_dir + name + '.pt'
        full_attention = torch.load(attent_path, map_location=self.device)
        attention_map = full_attention[11,:,:,:]
        attention_map = torch.sum(attention_map, 0)
        attention_map = padd_attent(attention_map)
        label = self.labels['norm_dG'].iloc[idx]
        #label = self.labels['dG'].iloc[idx]
        label = torch.tensor(label)
        if self.transform:
            attention_map = self.transform(attention_map)
        if self.target_transform:
            label = self.target_transform(label)
        return attention_map, label, name
    
class MultiAttentionDataset(Dataset):
    """ Keeps track of the the attentions and the normalized dG values."""
    def __init__(self, 
                    selection,
                    device, 
                    database_file = '/home/lucas/esmmsa/data/stability/ProTherm2.csv',
                    attent_dir = '/mnt/nasdata/lucas/data/row_attent/', 
                    transform=None, 
                    target_transform=None):
        
        self.database = pd.read_csv(database_file)
        self.database = self.database[self.database['MUTATION'] == 'WT']
        self.device = device
        self.labels = self.database.iloc[selection]
        self.attent_dir = attent_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        name = self.labels['UniProt_ID'].iloc[idx]
        attent_path = self.attent_dir + name + '_WT_.pt'
        try:
            full_attention = torch.load(attent_path, map_location=self.device)
            attention_map = full_attention[11,:,:,:]
            #attention_map = torch.sum(attention_map, 0)
            attention_map = padd_multi_attent(attention_map)
            label = self.labels['dG_minmax_scale'].iloc[idx]
            #label = self.labels['dG'].iloc[idx]
            label = torch.tensor(label)
            if self.transform:
                attention_map = self.transform(attention_map)
            if self.target_transform:
                label = self.target_transform(label)
        except:
            print(name)
        return attention_map, label, name
    
class testingMultiAttentionDataset(Dataset):
    """ Keeps track of the the attentions and the normalized dG values."""
    def __init__(self, 
                    selection,
                    device, 
                    database_file = '/home/lucas/esmmsa/data/stability/ProTherm2.csv',
                    attent_dir = '/mnt/nasdata/lucas/data/row_attent/', 
                    transform=None, 
                    target_transform=None):
        
        #temp = pd.read_csv(labels_file) #contains UniProtID and therm-data
        database = pd.read_csv(database_file)
        self.device = device
        database['keep'] = False
        for id in selection:
            match = database['UniProt_ID'] == id
            database['keep'] += match 
        database = database[database['MUTATION'] == 'WT']
        self.database = database[database['keep'] == True]
        #self.data = database.
        #self.labels = temp.iloc[selection]
        self.attent_dir = attent_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.database.shape[0]

    def __getitem__(self, idx):
        UniProtID = self.database['UniProt_ID'].iloc[idx]
        typ = self.database['MUTATION'].iloc[idx]
        name = UniProtID + '_' + typ 

        attent_path = self.attent_dir + UniProtID + '_' + typ + '_.pt'
        full_attention = torch.load(attent_path, map_location=self.device)
        attention_map = full_attention[11,:,:,:]
        attention_map = padd_multi_attent(attention_map)

        label = self.database['dG_minmax_scale'].iloc[idx]
        #label = self.labels['dG'].iloc[idx]
        label = torch.tensor(label)
        if self.transform:
            attention_map = self.transform(attention_map)
        if self.target_transform:
            label = self.target_transform(label)
        return attention_map, label, name

def padd_attent(attention_map):
    """ Padd attention maps to a size corresponding the map of an MSA of the size of MAX_MSA_COL_NUM x MAX_MSA_COL_NUM
        It is padded with np. and the rows and columns are extended after the obtained values.
    
    : parameter attention_map: the representations to embed
    : return: the padded attention_map, now at size [MAX_MSA_COL_NUM x MAX_MSA_COL_NUM]"""

    padd_value = 0
    MAX_MSA_COL_NUM = 512 - 1 
    length = attention_map.shape[0]
    numpad = (MAX_MSA_COL_NUM - length) // 2
    evenout = 0
    if length + (2 * numpad) != MAX_MSA_COL_NUM:
        evenout = 1
    padded_attention_map = torch.nn.functional.pad(input=attention_map,
                                                pad=(numpad,numpad + evenout, 
                                                     numpad, numpad + evenout),
                                                mode='constant',
                                                value=padd_value)
    return padded_attention_map

def padd_multi_attent(attention_map, side = 'both'):
    """ Padd attention maps to a size corresponding the map of an MSA of the size of MAX_MSA_COL_NUM x MAX_MSA_COL_NUM
        It is padded with 0. and the rows and columns are extended after the obtained values.
    
    : parameter attention_map: the representations to embed
    : return: the padded attention_map, now at size [MAX_MSA_COL_NUM x MAX_MSA_COL_NUM]"""

    padd_value = 0
    MAX_MSA_COL_NUM = 512 - 1 
    length = attention_map.shape[-1]
    if side == 'both':
        numpad = (MAX_MSA_COL_NUM - length) // 2
        evenout = 0
        if length + (2 * numpad) != MAX_MSA_COL_NUM:
            evenout = 1
        pad = [numpad, numpad + evenout, 
            numpad, numpad + evenout]
    if side == 'after':
        numpad = MAX_MSA_COL_NUM - length
        pad = [0, numpad, 0, numpad]

    for _ in range(attention_map.dim() - 2):
        pad.append(0)
        pad.append(0)
        

    padded_attention_map = torch.nn.functional.pad(input=attention_map,
                                                pad=pad,
                                                mode='constant',
                                                value=padd_value)
    return padded_attention_map

def padd_embed(embedding, side='both'):
    """ Padd embedding to a size corresponding the embedding of an MSA of the size of MAX_MSA_ROW_NUM x MAX_MSA_COL_NUM
    
    : parameter embedding: the representations to embed
    : return: the padded embedding, now at size [1, MAX_MSA_ROW_NUM, MAX_MSA_COL_NUM, 768]"""
    padd_value = alphabet.padding_idx
    MAX_MSA_ROW_NUM = 128  # 256
    MAX_MSA_COL_NUM = 512 - 1

    if side == 'both':
        if embedding.dim() == 2:
            length = embedding.shape[0]
            numpad = (MAX_MSA_COL_NUM - length) // 2
            evenout = 0
            if length + (2 * numpad) != MAX_MSA_COL_NUM:
                evenout = 1
            padded_embedding = torch.nn.functional.pad(input=embedding,
                                                        pad=(0, 0, numpad, numpad + evenout),
                                                        mode='constant',
                                                        value=padd_value)

        if embedding.dim() == 3:
            depth = embedding.shape[0]
            numpad = MAX_MSA_ROW_NUM - depth
            if numpad >= 0:
                padded_embedding = torch.nn.functional.pad(input=embedding,
                                                        pad=(0,0,0, 0, numpad, 0),
                                                        mode='constant',
                                                        value=padd_value)
            else:
                padded_embedding = embedding[:MAX_MSA_ROW_NUM,:]

            length = padded_embedding.shape[1]
            numpad = (MAX_MSA_COL_NUM - length) // 2
            evenout = 0
            if length + (2 * numpad) != MAX_MSA_COL_NUM:
                evenout = 1
            padded_embedding = torch.nn.functional.pad(input=padded_embedding,
                                                        pad=(0,0,numpad, numpad + evenout, 0, 0),
                                                        mode='constant',
                                                        value=padd_value)
            
    if side == 'after':
        if embedding.dim() == 2:
            length = embedding.shape[0]
            numpad = MAX_MSA_COL_NUM - length
            padded_embedding = torch.nn.functional.pad(input=embedding,
                                                        pad=(0, 0, 0, numpad),
                                                        mode='constant',
                                                        value=padd_value)

        if embedding.dim() == 3:
            depth = embedding.shape[0]
            numpad = MAX_MSA_ROW_NUM - depth
            if numpad >= 0:
                padded_embedding = torch.nn.functional.pad(input=embedding,
                                                        pad=(0,0,0, 0, numpad, 0),
                                                        mode='constant',
                                                        value=padd_value)
            else:
                padded_embedding = embedding[:MAX_MSA_ROW_NUM,:]

            length = padded_embedding.shape[1]
            numpad = MAX_MSA_COL_NUM - length
            padded_embedding = torch.nn.functional.pad(input=padded_embedding,
                                                        pad=(0,0,0, numpad, 0, 0),
                                                        mode='constant',
                                                        value=padd_value)
    
    return padded_embedding

def denormalize(result, mean, std):
    """ Get the actual value from a normalized value 
    : param result: the normalized value
    : param mean: the mean value used to normalize
    : param std: the standard deviation used to normalize
    : return: the actual value"""
    return result * std + mean

def inference (x, model) -> float:
    """Perform inference on a single embedding.
    : param x: non-padded embeddings
    : param model: the model to perform inference on
    : return: the predicted dG value from embeddings"""
    x = padd_embed(x)
    x = x[0,:,:]
    y = model(x)
    y = y.cpu().detach().numpy()[0,0]
    return y

def classify(x):
    if x > 0:
        return torch.Tensor([1, 0])
    else:
        return torch.Tensor([0, 1])

def numel(m: torch.nn.Module, only_trainable: bool = True):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)