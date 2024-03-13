from torch import nn
import torch
import math
from typing import List, Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2Block,
)


class Ttune_Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.config = config
        self.seq_len = config.seq_len
        self.rank = self.config.rank
        self.start_pos = config.seq_len - self.config.rank
        
        self.query_vector_0 = nn.Parameter(torch.ones(self.rank, config.hidden_size))
        self.query_vector_1 = nn.Parameter(torch.ones(config.hidden_size))
        
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
        temp_query = ()
        ############################   Ttune   ##########################
        if query.shape[-2] >= self.rank:
            q0 = query[:, self.start_pos:self.seq_len].mul(self.query_vector_0)
            q1 = query[:, :self.start_pos].mul(self.query_vector_1)
            query = torch.cat((q1, q0, query[:, self.seq_len:]), dim = -2)
        elif query.shape[-2] == 1:
            pass
        else:
            raise Exception("check sequence len")
        #################################################################
        
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            #print(query.shape)
            #print(attention_mask.shape)
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)        

    
    
class Ttune_Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.config = config
        self.seq_len = config.seq_len
        self.rank = self.config.rank
        self.start_pos = config.seq_len - self.config.rank
        
        self.attn = Ttune_Attention(config, layer_idx=layer_idx)
        if config.add_cross_attention:
            self.crossattention = Ttune_Attention(config, is_cross_attention=True, layer_idx=layer_idx)
        
        self.attention_vector_0 = nn.Parameter(torch.zeros(self.rank, config.hidden_size))
        self.attention_vector_1 = nn.Parameter(torch.zeros(config.hidden_size))
            
        self.hidden_vector_0 = nn.Parameter(torch.ones(self.rank, config.hidden_size))
        self.hidden_vector_1 = nn.Parameter(torch.ones(config.hidden_size))
        
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual
        
        
        
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights
            

        ###############################################################################################
        if hidden_states.shape[-2] >= self.rank:
            hidden_states[:, self.start_pos:self.seq_len] = torch.add(hidden_states[:, self.start_pos:self.seq_len], self.attention_vector_0)
            hidden_states[:, :self.start_pos] = torch.add(hidden_states[:, :self.start_pos], self.attention_vector_1)
        elif hidden_states.shape[-2] == 1:
            pass
        else:
            raise Exception("check sequence len") 
        ###############################################################################################
        
        
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
       
       
        temp_hidden = ()
        ###############################################################################################
        if hidden_states.shape[-2] >= self.rank:
            h0 = hidden_states[:, self.start_pos:self.seq_len].mul(self.hidden_vector_0)
            h1 = hidden_states[:, :self.start_pos].mul(self.hidden_vector_1)
            hidden_states = torch.cat((h1, h0, hidden_states[:, self.seq_len:]), dim = -2)
        elif hidden_states.shape[-2] == 1:
            pass
        else:
            raise Exception("check sequence len")
        ###############################################################################################
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)