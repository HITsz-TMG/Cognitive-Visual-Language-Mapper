import torch
import torch.nn as nn
import re
from typing import List, Optional, Tuple, Union
from transformers.models.blip_2 import Blip2QFormerModel, Blip2QFormerConfig
from local_transformers.transformers import OPTForCausalLM, OPTConfig, OPTModel


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_vision_qformer(config, delay_load=False, **kwargs):
    qformer_config = Blip2QFormerConfig(encoder_hidden_size=config.mm_hidden_size, hidden_size=config.mm_hidden_size,
                                            vocab_size=config.vocab_size, num_hidden_layers=6,
                                            num_attention_heads=16)
    
    knowledge_qformer = Blip2QFormerModel(qformer_config)
    
    return knowledge_qformer

def build_language_projector(config, delay_load=False, **kwargs):
    return nn.Linear(config.mm_hidden_size, config.hidden_size)

class KnowledgeVisionOPTProjector(nn.Module):
    def __init__(self, config, opt_model):
        super(KnowledgeVisionOPTProjector, self).__init__()
        self.config = config
        self.model = opt_model
        self.lm_hidden_size = config.opt_hidden_size
        
        self.query_tokens = nn.Parameter(torch.zeros(1, 32, config.opt_hidden_size))
        self.vision_projector = nn.Linear(config.mm_hidden_size, self.lm_hidden_size)
        self.language_projector = nn.Linear(self.lm_hidden_size, config.hidden_size)
        
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_features: Optional[torch.FloatTensor] = None,
    ):
        bsz = input_ids.size(0)
        image_bsz = image_features.size(0)
        
        assert image_bsz % bsz == 0
        input_ids = input_ids.repeat(image_bsz // bsz, 1)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(image_bsz // bsz, 1)
        
        encoder_hidden_states = self.vision_projector(image_features)
        
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        instruction_len = inputs_embeds.size(1)
        query_tokens_embeds = self.query_tokens.repeat(image_bsz, 1, 1)
        
        query_attention_mask = torch.ones(query_tokens_embeds.size()[:-1], dtype=torch.long,
                                          device=inputs_embeds.device)
        
        
        inputs_embeds = torch.concat([inputs_embeds, query_tokens_embeds], dim=1)
        attention_mask = torch.concat([attention_mask, query_attention_mask], dim=1)
        
                
        outputs = self.model.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                     encoder_hidden_states=encoder_hidden_states)[0]
        # print(instruction_len)
        # print(outputs.size())
        
        query_outputs = outputs[:, instruction_len:]
        
        knowledge_query_embeds = self.language_projector(query_outputs)
        
        return knowledge_query_embeds
        
    
def build_knowledge_opt_model(config, delay_load=False, **kwargs):
    opt_config = OPTConfig.from_pretrained(config.opt_model_path)
    opt_config.add_cross_attention = True
    opt_model =  OPTModel.from_pretrained(config.opt_model_path, config=opt_config)
    
    return KnowledgeVisionOPTProjector(config, opt_model)
    


