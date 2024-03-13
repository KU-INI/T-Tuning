from .ttune_block import Ttune_Block
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Model,
    GPT2LMHeadModel
)
from typing import Optional, Tuple, Union

class TtuneGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.h = nn.ModuleList([Ttune_Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
    
    
class TtuneGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = TtuneGPT2Model(config)
        self.seq_len = config.seq_len
        self.rank = config.rank