
import scanpy as sc 

adata = sc.read_h5ad('crc_res.h5ad')
slices = adata.obs['slice_id'].unique()
import numpy as np
import anndata as ad

adata_train = adata[adata.obs['slice_id'].isin(slices[:60])].copy()
adata_test = adata[adata.obs['slice_id'].isin(slices[60:])].copy()

adata_train.obsm['emd'] = np.load('emd_codex_crc_ct_0.1.npy')
adata_test.obsm['emd'] = np.load('emd_codex_crc_ct_0.1_new.npy')

adata_train.obsm['X_ct'] = np.load('emd_codex_crc_ct_0.1_b.npy')
adata_test.obsm['X_ct'] = np.load('emd_codex_crc_ct_0.1_new_b.npy')


final_adata = ad.concat([adata_train, adata_test], keys=['train', 'test'], label='train_test')

import faiss
import numpy as np
from scipy.stats import mode

def max_occurrence_in_string_rows(arr):
    most_frequent_values = []
    
    for row in arr:
        unique, counts = np.unique(row, return_counts=True)
        most_frequent_value = unique[np.argmax(counts)]
        most_frequent_values.append(most_frequent_value)
    
    return np.array(most_frequent_values)


full_adata = final_adata

for key in ['emd', 'X_ct']:

    # Sample data: an array of n vectors of dimension d
    #data_vectors = np.load('emd_codex_crc_ct_0.1.npy').astype('float32') 
    data_vectors = full_adata.obsm[key][full_adata.obs['train_test'] == 'train'] .astype('float32') 
    # Normalize the vectors to make cosine similarity equivalent to dot product
    faiss.normalize_L2(data_vectors)

    # Build the FAISS index
    dimension = data_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IP stands for inner product, equivalent to cosine similarity
    index.add(data_vectors)  # Add vectors to the index

    # Perform a similarity search
    #query_vector = np.load('emd_crc_codex_0.1_new.npy').astype('float32')  # Query vector
    query_vector = full_adata.obsm[key][full_adata.obs['train_test'] == 'test'] .astype('float32') 
    faiss.normalize_L2(query_vector)  # Normalize the query vector for cosine similarity



    for k in [1, 2, 3, 5]:

        distances, indices = index.search(query_vector, k)

        ref_result = adata_train.obs['neighborhood name'].values[indices]

        hit = 0
        total = 0

        for i, c in enumerate(adata_test.obs['neighborhood name'].values):
            if c in ref_result[i]:
                hit += 1
            total += 1

        print(f'{key} hit ratiao @ {k}: {hit/total}')



