import torch
import numpy as np
from torch import nn
from data import DuoLingo
from torch.utils.data import DataLoader, Subset
from model import Leitner, SM2
from tqdm import tqdm
import matplotlib.pyplot as plt


def spearman_rank(a, b):
    cov_ab = ((a - a.mean()) * (b - b.mean())).sum() / (a.shape[0] - 1)
    return cov_ab / (torch.std(a) * torch.std(b))

def evaluate(dataloader, model, device):
    a = []
    b = []
    with torch.no_grad():
        pbar = tqdm(total=len(dataloader))
        n = 0
        for i_batch, batch in enumerate(dataloader):

            # Transfer batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            delta = batch['delta']
            p = batch['p']

            # calculate theoretical target half life
            h = - delta / torch.log2(torch.clamp(p, 0.0 + 1e-7, 1.0 - 1e-7))
            # Run model with features
            h_theta = model(batch)
            a.append(h_theta)
            b.append(h)

            # Update progress bar
            pbar.update(1)
    
        pbar.close()

    a = torch.cat(a)
    b = torch.cat(b)
    return spearman_rank(a, b)
    

if __name__ == "__main__":
    dataset = DuoLingo()
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)
    dataset = Subset(dataset, idx[:1000000])
    dataloader = DataLoader(dataset, batch_size=100000, shuffle=True, num_workers=8, drop_last=True)

    print("Leitner Spearman Correlation: {}".format(evaluate(dataloader, Leitner(), 'cuda')))
    print("SM2 Spearman Correlation: {}".format(evaluate(dataloader, SM2(), 'cuda')))

