from torch_geometric.data import DataLoader,InMemoryDataset



import scanpy as sc
import anndata as ad 


import scanpy as sc
import numpy as np
import anndata as ad

from collections import defaultdict

import squidpy as sq

from torch_geometric.utils import k_hop_subgraph


import anndata
import squidpy as sq
import torch
from torch_geometric.data import Data, InMemoryDataset
import networkx as nx
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import networkx as nx

import torch_geometric

from concurrent.futures import ThreadPoolExecutor, as_completed

def add_attributes(dataset, train_indices, test_indices):
    data_list = []
    for i, data in enumerate(dataset):
        if i in train_indices:
            data.train_mask = True
            data.test_mask = False 
            data.val_mask = False
            data_list.append(data)
        elif i in test_indices:
            data.train_mask = False
            data.test_mask = True 
            data.val_mask = True
            data_list.append(data)
    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset

def extract_3_hop_subgraph_pyg(pyg_data, node_idx):
    subset, edge_index, _, _ = k_hop_subgraph(node_idx, 2, pyg_data.edge_index, relabel_nodes=True)
    subgraph = Data(x=pyg_data.x[subset], y=pyg_data.y[node_idx], edge_index=edge_index)
    return subgraph


def construct_encoder(adata_, lb_key, oh_key):

    lb = LabelEncoder()
    oh = OneHotEncoder()
    bt = LabelEncoder()

    lb.fit(adata_.obs[lb_key].astype(str).values)
    oh.fit(adata_.obs[oh_key].astype(str).values.reshape(-1, 1))
    bt.fit(adata_.obs['slice_id'].astype(str).values.reshape(-1, 1))

    return lb, oh, bt

from tqdm import tqdm

from scipy.sparse import issparse

class MEDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(MEDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['res.h5ad']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        adata_  = sc.read_h5ad('/home/ubuntu/yancui/MARVEL/adata/temp.h5ad')
        lb, oh, bt = construct_encoder(adata_, 'label', 'cell_type')
        self.feat_type = adata_.uns['feat_type']
        sq.gr.spatial_neighbors(adata_, library_key='slice_id')
        for s in tqdm(adata_.obs['slice_id'].unique(), desc="Processing slices"):
            adata = adata_[adata_.obs['slice_id'] == s].copy()
            if self.feat_type  == 'ct':
                feature = oh.transform(adata.obs['cell_type'].astype(str).values.reshape(-1,1)).toarray()
            else:
                feature = adata.X
            sq.gr.spatial_neighbors(adata)

            if issparse(feature):
                feature = feature.toarray()

            y_ = lb.transform(adata.obs['label'].astype(str).values)
            G = nx.from_numpy_array(adata.obsp['spatial_connectivities'].toarray())
            data = torch_geometric.utils.from_networkx(G)
            x = torch.from_numpy(feature).float()
            y = torch.from_numpy(np.array(y_))
            data.x = x 
            data.y = y

            # Parallel processing for extracting 3-hop subgraphs
            with ThreadPoolExecutor(max_workers=32) as executor:  # Adjust max_workers as needed
                futures = [executor.submit(extract_3_hop_subgraph_pyg, data, i) for i in range(len(adata))]
                
                # Collect the subgraphs as they are processed
                for future in tqdm(as_completed(futures), total=len(adata), desc="Processing nodes"):
                    three_hop_subgraph = future.result()
                    data_list.append(three_hop_subgraph)
        
        self.save(data_list, self.processed_paths[0])
  
