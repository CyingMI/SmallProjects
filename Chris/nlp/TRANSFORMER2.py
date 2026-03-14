import torch
from torch import nn
import torch.nn.functional as f
import math

class transformer(nn.Module):
    def __init__(self,
                 src_pad_ix,
                 trg_pad_idx,
                 enc_voc_size,
                 dec_voc_size,
                 embedding_dim,
                 n_heads,
                 ffn_hidden,
                 n_layers,
                 drop_prob,
                 device):
        super(transformer,self).__init__()
        self.encoder=Encoder