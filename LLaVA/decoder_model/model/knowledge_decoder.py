import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from local_transformers.transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from local_transformers.transformers import OPTForCausalLM, OPTConfig, OPTModel
from local_transformers.transformers import CLIPModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union
from local_transformers.transformers.configuration_utils import PretrainedConfig

class KnowledgeVisionGPT2Config(PretrainedConfig):
    def __init__(
        self,
        gpt2_model_path="/data/share/Model/gpt2-large/",
        clip_model_path="/data/cxy/LLaVA/local_models/clip-vit-large-patch14-336",
        select_layer=-2
    ):
        self.hidden_sizes = [1024, 1280]
        self.gpt_model_path = gpt2_model_path
        self.clip_model_path = clip_model_path
        self.select_layer = select_layer

class KnowledgeVisionGPT2(nn.Module):
    def __init__(self, config: KnowledgeVisionGPT2Config):
        super(KnowledgeVisionGPT2, self).__init__()
        
        self.config = config
        
        gpt2_config = GPT2Config.from_pretrained(config.gpt_model_path)
        gpt2_config.add_cross_attention=True
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(config.gpt_model_path, config=gpt2_config)
        self.clip_model = CLIPModel.from_pretrained(config.clip_model_path).vision_model
        
        self.vision_hidden_size = self.clip_model.config.hidden_size
        self.lm_hidden_size = self.gpt2_model.config.hidden_size
        
        self.vision_projector = nn.Linear(self.vision_hidden_size, self.lm_hidden_size)
        
        self.select_layer = config.select_layer
        # self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        images: torch.Tensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        image_forward_outs = self.clip_model(pixel_values=images, output_hidden_states=True)        
        image_features = image_forward_outs.hidden_states[self.select_layer]
        
        encoder_hidden_states = self.vision_projector(image_features)
        
        outputs = self.gpt2_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                  encoder_hidden_states=encoder_hidden_states, labels=labels,
                                  output_attentions=output_attentions, output_hidden_states=output_hidden_states,)
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        images: torch.Tensor = None,
        **kwargs
    ):
        image_forward_outs = self.clip_model(pixel_values=images, output_hidden_states=True) 
        
        
class KnowledgeVisionOPTConfig(OPTConfig):
    def __init__(
        self,
        clip_model_path="/data/cxy/LLaVA/local_models/clip-vit-large-patch14-336",
        select_layer=-2,
        **kwargs
    ):
        super().__init__(**kwargs,)
        self.clip_model_path = clip_model_path
        self.select_layer = select_layer
        self.add_cross_attention = True
        
    

class KnowledgeVisionOPT(OPTForCausalLM):
    def __init__(self, config: KnowledgeVisionOPTConfig):
        super(KnowledgeVisionOPT, self).__init__(config)
        
        self.config = config
        
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        
        self.model = OPTModel(config)
        self.clip_model = CLIPModel.from_pretrained(config.clip_model_path).vision_model

        self.lm_hidden_size = config.hidden_size
        self.vision_hidden_size = self.clip_model.config.hidden_size
        
        self.vision_projector = nn.Linear(self.vision_hidden_size, self.lm_hidden_size)
        
        self.select_layer = config.select_layer
        
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        images: torch.Tensor = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        image_forward_outs = self.clip_model(pixel_values=images, output_hidden_states=True)        
        image_features = image_forward_outs.hidden_states[self.select_layer]
        
        encoder_hidden_states = self.vision_projector(image_features)
        
        outputs = self.model.decoder(input_ids=input_ids, attention_mask=attention_mask,
                                     head_mask=head_mask, past_key_values=past_key_values,
                                     inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                                     use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states,)
        
        logits = self.lm_head(outputs[0]).contiguous()
        
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.pop("images", None)
            }
        )
        return model_inputs
    
