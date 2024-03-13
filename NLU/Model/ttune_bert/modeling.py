from .ttune_encoder import Ttune_Encoder
import torch

from transformers.models.bert.modeling_bert import (
    BertModel,
    BertForSequenceClassification
)


class TtuneBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = Ttune_Encoder(config)
        
class TtuneBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = TtuneBertModel(config)