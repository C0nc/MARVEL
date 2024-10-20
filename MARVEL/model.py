import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator, RFEvaluator, LREvaluator, from_predefined_split
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool, GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.data import DataLoader

import anndata
import squidpy as sq
import torch
from torch_geometric.data import Data, InMemoryDataset
import networkx as nx
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import networkx as nx

from .data import  add_attributes


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        return x

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


from tqdm import tqdm


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_cov=2):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_mean_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


def train(encoder_model, contrast_model, classifier, dataloader, label_dataloader, encoder_optimizer, device):
    encoder_model.train()
    epoch_loss = 0
    for i, (data, data_l) in enumerate(zip(dataloader, label_dataloader)):
        data = data.to(device)
        data_l = data_l.to(device)
        #data.x =  F.normalize(data.x, p=2, dim=-1)

        if data.x is None:
            num_nodes = data.batch.size(0)

            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, g, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)

        if classifier is not None:
            _, g_l, _, _, _, _ = encoder_model(data_l.x, data_l.edge_index, data_l.batch)

            pred = classifier(g_l)

            target = data_l.y
    
            cls_loss = F.cross_entropy(pred, target)


        #encoder_model.train()
        #encoder_optimizer.zero_grad()
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]

        # compute extra pos and neg masks for semi-supervised learning
        extra_pos_mask = torch.eq(data.y, data.y.unsqueeze(dim=1)).to(device)
        #print(extra_pos_mask.shape)
        # construct extra supervision signals for only training samples
        extra_pos_mask[~data.train_mask][:, ~data.train_mask] = False
        extra_pos_mask.fill_diagonal_(False)
        # pos_mask: [N, 2N] for both inter-view and intra-view samples
        extra_pos_mask = torch.cat([extra_pos_mask, extra_pos_mask], dim=1).to(device)
        # fill interview positives only; pos_mask for intraview samples should have zeros in diagonal
        extra_pos_mask.fill_diagonal_(True)

        extra_neg_mask = torch.ne(data.y, data.y.unsqueeze(dim=1)).to(device)
        extra_neg_mask[~data.train_mask][:, ~data.train_mask] = True
        extra_neg_mask.fill_diagonal_(False)
        extra_neg_mask = torch.cat([extra_neg_mask, extra_neg_mask], dim=1).to(device)

        #print(extra_neg_mask.shape, extra_pos_mask.shape)
        if classifier is not None:
            loss = cls_loss
        else:
            loss = contrast_model(g1=g1, g2=g2,  extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)         

        encoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()

        
        epoch_loss += loss.item()

    return epoch_loss


from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


def test(encoder_model, cls, dataloader, train_indices, test_indices, device, eval=True):
    encoder_model.eval()
    x = []
    y = []
    preds = []
    for data in dataloader:
        data = data.to(device)
        #data.x =  F.normalize(data.x, p=2, dim=-1)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _ =  encoder_model(data.x, data.edge_index, data.batch)
        if cls is not None:
            pred = cls(g)
            preds.append(pred.argmax(dim=1).detach())
        x.append(g.detach())

        y.append(data.y.detach())

    x = torch.cat(x, dim=0)

    y = torch.cat(y, dim=0)

    if cls is not None:
        preds = torch.cat(preds, dim=0).cpu()
        y = y.cpu()
        result = {'micro_f1':f1_score(preds.numpy(), y.numpy(), average='micro') , 'macro_f1': f1_score(preds.numpy(), y.numpy(), average='macro')}
    else:
        result = LREvaluator()(x, y, {'train': train_indices, 'test': test_indices, 'valid': test_indices})
        result['pred'] = np.ones(x.shape[0])
        result['pred'][train_indices] = y.cpu().numpy()[train_indices]
        result['pred'][test_indices] = result['res']
        result['indices'] = np.zeros(x.shape[0]) 
        result['indices'][train_indices] = 1 
        
    return result, x




from sklearn.model_selection import train_test_split

class MARVEL(object):
    def __init__(self, dataset, train_indices, test_indices, batch_size=4096, hidden_dim=128, num_layer=2, classifier=None, device='cuda'):
        self.train_indices, self.test_indices = train_indices, test_indices
        self.dataset = add_attributes(dataset, train_indices, test_indices)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        self.label_dataloader = DataLoader(dataset[train_indices], batch_size=batch_size, shuffle=True, drop_last=False)
        self.test_dataloader = DataLoader(dataset[test_indices], batch_size=batch_size, shuffle=False, drop_last=False)
        input_dim = max(dataset.num_features, 1)
        aug1 = A.Identity()
        aug2 = A.RandomChoice([A.RWSampling(num_seeds=42, walk_length=10),
                            A.NodeDropping(pn=0.2),
                            #A.FeatureMasking(pf=0.4),
                            A.EdgeRemoving(pe=0.2),
                            ], 1)
        gconv = GConv(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layer).to(device)
        encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
        self.classifier = classifier
        self.encoder_model = encoder_model
        self.device = device
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(0.2), mode='G2G', intraview_negs=True).to(device)
        if self.classifier is not None:
            self.encoder_optimizer = Adam(list(encoder_model.parameters()) + list(classifier.parameters()), lr=1e-3, weight_decay=1e-4)
        else:
            self.encoder_optimizer = Adam(list(encoder_model.parameters()), lr=1e-3, weight_decay=1e-4)


    def fit(self, epoch=100):
        with tqdm(total=epoch, desc='(T)') as pbar:
            for epoch in range(1, epoch+1):
                loss = train(self.encoder_model, self.contrast_model, self.classifier, self.dataloader, self.label_dataloader, self.encoder_optimizer, self.device)
                pbar.set_postfix({'loss': loss})
                pbar.update()

    def evaluate(self, eval_batch_size=4096):
        dataloader = DataLoader(self.dataset, batch_size=eval_batch_size, shuffle=False)
        if self.classifier is None:
            test_result, emd = test(self.encoder_model, self.classifier, dataloader, self.train_indices, self.test_indices, self.device)
        else:
            test_result, emd = test(self.encoder_model, self.classifier, self.test_dataloader, self.train_indices, self.test_indices, self.device)

        return test_result, emd

    def extract_emd(self, dataset, eval_batch_size=4096):

        dataloader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
        new_reps = []

        self.encoder_model.eval()
        for data in dataloader:
            data = data.to('cuda')
            _, g = self.encoder_model.encoder(data.x, data.edge_index, data.batch)
            new_reps.append(g.detach().cpu())

        new_reps = torch.concat(new_reps).numpy()

        return new_reps
    
    def load(self, path):
        self.encoder_model.encoder = torch.load(path)
    
    def save(self, path):
        torch.save(self.encoder_model.encoder, path)

