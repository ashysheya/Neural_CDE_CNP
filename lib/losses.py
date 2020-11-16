import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from lib.utils import device

def get_loss(args):
    return {'nll': GaussianLogLikelihoodLoss(), 
                'mse': MSELoss()}
    

class GaussianLogLikelihoodLoss(nn.Module):
    """Gaussian Log-likelihood."""
    def __init__(self):
        super(GaussianLogLikelihoodLoss, self).__init__()

    def forward(self, gt, pred, start, end, mode='mean'):
        gaussian = Normal(loc=pred['mean_pred'][:, start:end], 
                          scale=pred['std_pred'][:, start:end])

        log_prob = gaussian.log_prob(gt['y'][:, start:end])
        if mode == 'mean':
            log_prob = ((log_prob * gt['mask_y'][:, start:end]).sum()/ \
                gt['mask_y'][:, start:end].sum())           
            return -log_prob
        elif mode == 'batched_mean':
            log_prob = (log_prob * gt['mask_y'][:, start:end]).sum(dim=(1, 2))
            return -log_prob.mean()


class MSELoss(nn.Module):
    """Mean squarred error loss."""
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, gt, pred, start, end, mode='mean'):
        mse_loss = gt['mask_y'][:, start:end] * (
            gt['y'][:, start:end] - pred['mean_pred'][:, start:end])**2

        if mode == 'mean':
            return mse_loss.sum()/gt['mask_y'][:, start:end].sum()
        elif mode == 'batched_mean':
            return mse_loss.sum(dim=(1, 2)).mean()


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, pred, mode='mean'):
        kl = -torch.log(pred['std_z0']) + pred['std_z0']**2/2
        kl += pred['mean_z0']**2/2 - 0.5
        if mode == 'mean':
            return kl.mean()
        elif mode == 'batched_mean':
            return kl.sum(dim=-1).mean()
