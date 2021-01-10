import torch
from torch import nn
from data import DuoLingo
from torch.utils.data import DataLoader
from model import Leitner, SM2
from tqdm import tqdm
# Mean Absolute Error
def mean_absolute_error(p, p_theta):
    return nn.L1Loss(p, p_theta)

def evaluate(dataloader, model, device):

  
    loss_sum = 0
    with torch.no_grad():
        pbar = tqdm(total=len(dataloader))
        n = 0
        for i_batch, batch in enumerate(dataloader):
            # Get features from batch
            correct = batch['correct'].to(device)
            incorrect = batch['incorrect'].to(device)
            delta = batch['delta'].to(device)
            p = batch['p'].to(device)
            hist_p = batch['hist_p'].to(device)

            # calculate theoretical target half life
            h = delta / torch.log2(p+1e-7)

            # Run model with features
            h_theta = model(hist_p)

            # Calculate predicted recall from model half life
            p_theta = torch.exp2(-(delta.double()/h_theta))
            # print(torch.isnan(p_theta).any())
            loss = torch.mean(torch.abs(p - p_theta))
            # print(torch.abs(p - p_theta))
            # Compute loss
            loss_sum = (n*loss_sum + loss) / (n+1)

            # Update progress bar
            pbar.update(1)
            pbar.set_description("Loss: {}".format(loss_sum.item()))
            n+=1

    
        pbar.close()


    print(loss_sum/len(dataloader))




if __name__ == "__main__":
    dataset = DuoLingo()
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=8, drop_last=True)
    model = SM2()
    evaluate(dataloader, model, 'cuda')

