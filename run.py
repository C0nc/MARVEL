import torch

from MARVEL.model import MARVEL, Classifier

import numpy as np

from MARVEL.utils import seed_everything, baseline, calculate_class_proportions, split

from MARVEL.data import MEDataset

from sklearn.model_selection import train_test_split
import scanpy as sc 
import anndata as ad


def main(args):
    seed_everything()
    full_adata = sc.read_h5ad('crc_res.h5ad')
    adata_list = []
    new_adata_list = []
    train_adata_list = []
    for s in full_adata.obs['slice_id'].unique():
        adata = full_adata[full_adata.obs['slice_id'] == s].copy()
        if (adata.obs['neighborhood name'] == 'Follicle').sum() == 0:
            train_adata_list.append(adata)
        else:
            new_adata_list.append(adata)
        adata_list.append(adata)
    
    if args.novel:
        args.dataset = args.dataset + '_novel'

    if args.novel:
        adata = ad.concat(train_adata_list)

        adata.obs['cell_type'] = adata.obs['ClusterName'].values

        adata.obs['label'] = adata.obs['neighborhood name'].values

    else:
        adata = ad.concat(adata_list[:60])

        adata.obs['cell_type'] = adata.obs['ClusterName'].values

        adata.obs['label'] = adata.obs['neighborhood name'].values

    if 'ct' in args.dataset:
        adata.uns['feat_type'] = 'ct'
    else:
        adata.uns['feat_type'] = 'exp'
    adata.write_h5ad('MARVEL/adata/temp.h5ad')

    device = torch.device(f'cuda:{args.gpu}')
    dataset = MEDataset(root=f'{args.dataset}_train_1')

    if args.novel:
        adata = ad.concat(new_adata_list)

        adata.obs['cell_type'] = adata.obs['ClusterName'].values

        adata.obs['label'] = adata.obs['neighborhood name'].values

    else:
        adata = ad.concat(adata_list[60:])

        adata.obs['cell_type'] = adata.obs['ClusterName'].values

        adata.obs['label'] = adata.obs['neighborhood name'].values

    adata.obs['cell_type'] = adata.obs['ClusterName'].values

    adata.obs['label'] = adata.obs['neighborhood name'].values

    if 'ct' in args.dataset:
        adata.uns['feat_type'] = 'ct'
    else:
        adata.uns['feat_type'] = 'exp'

    adata.write_h5ad('MARVEL/adata/temp.h5ad')

    test_dataset = MEDataset(root=f'{args.dataset}_test_1')

    labels = dataset.y.numpy()

    test_indices, train_indices = train_test_split(
        np.arange(len(labels)),
        test_size=args.proportion,
        stratify=labels,
        random_state=42  # Ensure reproducibility
    )

    if args.baseline:
        if 'ct' in args.dataset:
            X =  []
            y = []
            for data in dataset:
                X.append(calculate_class_proportions(data.x.numpy()))
                y.append(data.y)
            X = np.stack(X)
            y= np.stack(y)
            np.save(f'emd_{args.dataset}_{args.proportion}_b.npy', X)   
        else:
            X =  []
            y = []
            for data in dataset:
                X.append(data.x.mean(axis=0))
                y.append(data.y)
            X = np.stack(X)
            y= np.stack(y)
            np.save(f'emd_{args.dataset}_{args.proportion}_b.npy', X)   
        f1_rf_micro, f1_rf_macro, f1_lr_micro, f1_rf_macro  = baseline(X[train_indices], X[test_indices], y[train_indices], y[test_indices])
        np.save(f'res_{args.dataset}_{args.proportion}.npy', np.array([f1_rf_micro, f1_rf_macro, f1_lr_micro, f1_rf_macro]))   
        if 'ct' in args.dataset:
            X =  []
            y = []
            for data in test_dataset:
                X.append(calculate_class_proportions(data.x.numpy()))
                y.append(data.y)
            X = np.stack(X)
            y= np.stack(y)
            np.save(f'emd_{args.dataset}_{args.proportion}_new_b.npy', X)   
        else:
            X =  []
            y = []
            for data in test_dataset:
                X.append(data.x.mean(axis=0))
                y.append(data.y)
            X = np.stack(X)
            y= np.stack(y)
            np.save(f'emd_{args.dataset}_{args.proportion}_new_b.npy', X)   

    else:
        if args.cls:
            classifier = Classifier(128*2, 64,  dataset.num_classes).to(device)
            model = MARVEL(dataset, train_indices, test_indices, device=device, classifier=classifier)
        else:
            model = MARVEL(dataset, train_indices, test_indices, device=device)
        model.fit(epoch=100)
        test_result, _ = model.evaluate()
        train_emd = model.extract_emd(dataset)
        test_emd =  model.extract_emd(test_dataset)
        np.save(f'emd_{args.dataset}_{args.proportion}_new.npy', test_emd)  
        np.save(f'emd_{args.dataset}_{args.proportion}.npy', train_emd)  
        #model.save('test.pt')
        print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
        #np.save(f'our_res_{args.dataset}_{args.proportion}_pred.npy', test_result['pred'])   
        #np.save(f'our_res_{args.dataset}_{args.proportion}_index.npy', test_result['indices']) 
        #np.save(f'our_res_{args.dataset}_{args.proportion}.npy', np.array([test_result['micro_f1'], test_result['macro_f1']]))   


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for dataset processing.")

    parser.add_argument(
        '--dataset',
        type=str,
        default='codex_crc_ct' ,
        help='Path to the dataset file (default: default_dataset.csv)'
    )

    parser.add_argument(
        '--cls',
        default=False,
        action='store_true',
    )

    parser.add_argument(
        '--proportion',
        type=float,
        default=0.1,
        help='Proportion of the dataset to use (default: 1.0, i.e., 100%)'
    )


    parser.add_argument(
        '--novel',
        default=False,
        action='store_true',
    )


    parser.add_argument(
        '--baseline',
        default=False,
        action='store_true',
        help='Proportion of the dataset to use (default: 1.0, i.e., 100%)'
    )


    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='Proportion of the dataset to use (default: 1.0, i.e., 100%)'
    )



    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)