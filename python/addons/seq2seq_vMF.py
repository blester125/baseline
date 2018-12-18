import math
import numpy as np
from scipy.special import iv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from baseline.model import register_model
from baseline.pytorch.seq2seq.model import Seq2SeqModel


@register_model(task='seq2seq', name='vMF')
class Seq2SeqvMF(Seq2SeqModel):
    def __init__(self, *args, **kwargs):
        super(Seq2SeqvMF, self).__init__(*args, **kwargs)

    def create_loss(self):
        return vMFCriterion()


class vMFCriterion(nn.Module):
    def __init__(self, reg1=0.2, reg2=0.1):
        self.reg1 = reg1
        self.ref2 = reg2

    def __call__(self, pred, target):
        print(pred.shape)
        m = pred.size()[1]
        norm_pred = torch.norm(pred, dim=1)
        norm = -approx_log_C_m_K(m, norm_pred)
        sim = self.reg2 * torch.matmul(pred.unsqueeze(1), target.unsqueeze(-1)).squeeze()
        reg = self.reg1 * norm_pred
        return torch.mean(norm - sim + reg)


def approx_log_C_m_K(m, K):
    """This is a approximation of logC_m(K).

    This approx is based of the integral of the approx of the gradient
    so it may be off by some constant C. This should be ok for optimization
    """
    k_square = torch.pow(K, 2)
    m_1_square = torch.pow(m + 1, 2).to(K.dtype)
    sqrt_m_k = torch.sqrt(m_1_square + k_square)
    m_minus_1 = (m - 1).to(K.dtype)
    return sqrt_m_k - m_minus_1 * torch.log(m_minus_1 + sqrt_m_k)


def greedy_closest(query, keys):
    # Query: B, E
    # Keys: V, E
    sim = torch.matmul(query, keys.transpose(0, 1)) # B, V
    print(sim)
    _, best = torch.max(sim, dim=1) # B
    return best

def normalize(x, dim=1):
    norm = torch.norm(x, dim=dim).unsqueeze(dim)
    return x / norm
