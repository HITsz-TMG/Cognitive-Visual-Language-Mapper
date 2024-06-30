import re
import os
import warnings
import shutil
import torch
from transformers import CLIPImageProcessor
from transformers import AutoTokenizer
from decoder_model.model.knowledge_decoder import KnowledgeVisionOPTConfig, KnowledgeVisionOPT

def load_pretrained_model(model_path, model_base, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    config = KnowledgeVisionOPTConfig.from_pretrained(model_base)
    model = KnowledgeVisionOPT.from_pretrained(model_base, config=config)
    
    image_processor = CLIPImageProcessor.from_pretrained(config.clip_model_path)
    
    knowledge_weights = torch.load(model_path, map_location="cpu")
    knowledge_weights = {(k[11:] if k.startswith('base_model.') else k): v for k, v in knowledge_weights.items()}
    if any(k.startswith('model.model.') for k in knowledge_weights):
        knowledge_weights = {(k[6:] if k.startswith('model.') else k): v for k, v in knowledge_weights.items()}
    model.load_state_dict(knowledge_weights, strict=False)
    
    return tokenizer, model, image_processor
    
    
    # config = KnowledgeVisionGPT2Config()
    # model = KnowledgeVisionGPT2(config)
    
    