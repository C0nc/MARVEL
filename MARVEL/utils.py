import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score

from GCL.eval import BaseEvaluator


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict, eval: bool = False):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')

                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch

                    pbar.set_postfix({'best test F1Mi': best_test_micro, 'F1Ma': best_test_macro})
                    pbar.update(self.test_interval)

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro,
            'res': y_pred 
        }



from sklearn.model_selection import train_test_split

import random
import os


import numpy as np



def calculate_class_proportions(one_hot_array):
    # Sum the one-hot vectors along the first axis to count the occurrences of each class
    class_counts = np.sum(one_hot_array, axis=0)
    
    # Calculate the total number of samples
    total_samples = np.sum(class_counts)
    
    # Calculate the proportion of each class
    class_proportions = class_counts / total_samples
    
    return class_proportions


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


import numpy as np

def baseline(X_train, X_test, y_train, y_test):



    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)

    # Logistic Regression
    logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
    logistic_regression.fit(X_train, y_train)
    y_pred_lr = logistic_regression.predict(X_test)

    # Calculate F1 Scores
    f1_rf_macro = f1_score(y_test, y_pred_rf, average='macro')
    f1_rf_micro = f1_score(y_test, y_pred_rf, average='micro')

    f1_lr_macro = f1_score(y_test, y_pred_lr, average='macro')
    f1_lr_micro = f1_score(y_test, y_pred_lr, average='micro')

    print(f"Random Forest F1 Macro: {f1_rf_macro:.4f}")
    print(f"Random Forest F1 Micro: {f1_rf_micro:.4f}")
    print(f"Random Forest Acc {accuracy_score(y_test, y_pred_lr):.4f}")
    
    print(f"Logistic Regression F1 Macro: {f1_lr_macro:.4f}")
    print(f"Logistic Regression F1 Micro: {f1_lr_micro:.4f}")
    print(f"Random Forest Acc: {accuracy_score(y_test, y_pred_rf):.4f}")
    
    return f1_rf_micro, f1_rf_macro, f1_lr_micro, f1_rf_macro 


def split(dataset, proportion):
    labels = dataset.y.numpy()

    test_indices, train_indices = train_test_split(
        np.arange(len(labels)),
        test_size=proportion,
        stratify=labels,
        random_state=42  # Ensure reproducibility
    )

    return train_indices, test_indices