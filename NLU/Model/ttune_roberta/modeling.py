from .ttune_encoder import Ttune_Encoder
import torch

from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaForSequenceClassification
)


class TtuneRobertaModel(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = Ttune_Encoder(config)
        
class TtuneRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = TtuneRobertaModel(config)