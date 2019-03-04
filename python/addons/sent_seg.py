import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline.pytorch.embeddings import LookupTableEmbeddings, register_embeddings


@register_embeddings(name='continuous')
class PassthroughEmbeddings(PytorchEmbeddings):
    def __init__(self, _, **kwargs):
        super(FeatureEmbeddings, self).__init__()


class SentSegTagger(nn.Module):
    def __init__(self, nchars, emb_dim, hidden_dim, dropout, feat_dim):
        super().__init__()
