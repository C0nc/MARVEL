
import numpy as np
import anndata as ad 

import scanpy as sc 
adata = sc.read_h5ad('crc_res.h5ad')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

adata_train = adata[adata.obs['train'] == 'train'].copy()
adata_test = adata[adata.obs['train'] != 'train'].copy()

adata_train.obsm['MARVEL'] = np.load('emd_codex_crc_ct_novel_0.1.npy')
adata_test.obsm['MARVEL'] = np.load('emd_codex_crc_ct_novel_new_0.1.npy')

adata_train.obsm['Cell Type Baseline'] = np.load('emd_codex_crc_ct_novel_0.1_b.npy')
adata_test.obsm['Cell Type Baseline'] = np.load('emd_codex_crc_ct_novel_0.1_new_b.npy')


final_adata = ad.concat([adata_train, adata_test], keys=['train', 'test'], label='train_test')

import faiss
import numpy as np
from scipy.stats import mode

from sklearn.metrics import roc_auc_score, average_precision_score

def max_occurrence_in_string_rows(arr):
    most_frequent_values = []
    
    for row in arr:
        unique, counts = np.unique(row, return_counts=True)
        most_frequent_value = unique[np.argmax(counts)]
        most_frequent_values.append(most_frequent_value)
    
    return np.array(most_frequent_values)



embedding_methods = ['MARVEL', 'Cell Type Baseline']

full_adata = final_adata

scores = {}

for key in ['MARVEL', 'Cell Type Baseline']:

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




    distances, indices = index.search(query_vector, 2)



    ref_result = (adata_test.obs['neighborhood name'] == 'Follicle').astype(int).values

    score = 1 - np.max(distances, axis=1)

    scores[key] = score

    print(f'{key} roc_auc: ', roc_auc_score(ref_result, score), 'roc_auprc: ', average_precision_score(ref_result, score))


plt.figure(figsize=(12, 6))

colors = ['#3D5C6F', '#E47159', '#F9AE78']

linestyle = ['-', '--', ':']

plt.subplot(1, 2, 1)
for i, method in enumerate(embedding_methods):
    fpr, tpr, _ = roc_curve(ref_result, scores[method])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{method} (AUC = {roc_auc:.4f})', color=colors[i], linestyle=linestyle[i])

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Different Embeddings')
plt.legend(loc="lower right")

# Subplot for PRC curves
plt.subplot(1, 2, 2)
for i, method in enumerate(embedding_methods):
    precision, recall, _ = precision_recall_curve(ref_result, scores[method])
    prc_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{method} (AUC = {prc_auc:.4f})', color=colors[i], linestyle=linestyle[i])

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Different Embeddings')
plt.legend(loc="lower right")
plt.grid(False)

#plt.tight_layout()
#plt.show()

plt.tight_layout()
plt.savefig(f'curve.png', bbox_inches='tight', pad_inches=0, format='png')
