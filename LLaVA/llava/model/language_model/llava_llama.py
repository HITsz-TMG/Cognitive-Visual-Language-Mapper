#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from llava.constants import KNOWLEDGE_QUERY_TOKEN_INDEX, DEFAULT_KNOWLEDGE_QUERY_TOKEN
from transformers import Blip2QFormerConfig, Blip2QFormerModel
from local_transformers.transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from local_transformers.transformers.models.instructblip import InstructBlipQFormerConfig, InstructBlipQFormerModel

from transformers.modeling_outputs import CausalLMOutputWithPast
IMGD_TOKEN_INDEX = 32000
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, LlavaKnowledgeMetaModel, LlavaKnowledgeMetaForCausalLM, LlavaOPTAttentionKnowledgeMetaForCausalLM, \
                        LlavaOPTKnowledgeMetaModel, LlavaOPTKnowledgeMetaForCausalLM, LlavaSAMOPTAttentionKnowledgeMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava"

class KnowledgeLlavaConfig(LlamaConfig):
    model_type = "llava"
    def __init__(self, opt_model_path="/data/share/Model/opt-1.3b", opt_hidden_size=2048,  **kwargs):
        super().__init__(**kwargs,)
        self.opt_model_path = opt_model_path
        self.opt_hidden_size = opt_hidden_size


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mm_hidden_size = 1024
        self.query_len = 16
        self.lmeye_query_tokens = nn.Parameter(torch.zeros(16, config.hidden_size))
        self.lmeye_query_projector = nn.Linear(config.hidden_size, self.mm_hidden_size)
        self.lmeye_language_projector = nn.Linear(self.mm_hidden_size, config.hidden_size)
        qformer_config = Blip2QFormerConfig(encoder_hidden_size=self.mm_hidden_size, hidden_size=self.mm_hidden_size,
                                            vocab_size=self.config.vocab_size, num_hidden_layers=4,
                                            num_attention_heads=16)
        self.lmeye_interactor = Blip2QFormerModel(qformer_config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        bsz = input_ids.size(0)
        image_len = self.get_vision_tower().num_patches - 1
        input_position_ids = torch.concat([torch.ones(bsz, image_len, device=input_ids.device), input_ids], dim=1)

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_embeds
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )
        query_tokens = self.lmeye_query_tokens.repeat(bsz, 1)
        inputs_embeds[input_position_ids == IMGD_TOKEN_INDEX] = query_tokens
        
        with torch.no_grad():
            query_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).last_hidden_state
        
        query_embeds = query_output[input_position_ids == IMGD_TOKEN_INDEX]
        
        query_tokens = self.lmeye_query_projector(query_embeds).view(-1, self.query_len, self.mm_hidden_size)
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_outputs = self.lmeye_interactor(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_output = query_outputs.last_hidden_state
        query_image_features = self.lmeye_language_projector(query_output).view(-1, self.config.hidden_size)
        
        inputs_embeds[input_position_ids == IMGD_TOKEN_INDEX] = query_image_features
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs
    
    
class KnowledgeLlavaLlamaModel(LlavaKnowledgeMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(KnowledgeLlavaLlamaModel, self).__init__(config)


class KnowledgeLlavaLlamaForCausalLM(LlamaForCausalLM, LlavaKnowledgeMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(KnowledgeLlavaLlamaForCausalLM, self).__init__(config)
        self.model = KnowledgeLlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.query_tokens = nn.Parameter(torch.zeros(1, 128, 1024))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # self.query_tokens = 
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        opt_input_ids: Optional[torch.LongTensor] = None,
        opt_attention_mask: Optional[torch.Tensor] = None,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        query_tokens = self.query_tokens.expand(input_ids.shape[0], -1, -1)
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                query_tokens
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs
    
class KnowledgeOPTLlavaLlamaModel(LlavaOPTKnowledgeMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(KnowledgeOPTLlavaLlamaModel, self).__init__(config)
        
class KnowledgeOPTLlavaLlamaForCausalLM(LlamaForCausalLM, LlavaOPTKnowledgeMetaForCausalLM):
    config_class = KnowledgeLlavaConfig

    def __init__(self, config):
        super(KnowledgeOPTLlavaLlamaForCausalLM, self).__init__(config)
        self.model = KnowledgeOPTLlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        opt_input_ids: Optional[torch.LongTensor] = None,
        opt_attention_mask: Optional[torch.Tensor] = None,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print(input_ids.size())
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                opt_input_ids,
                opt_attention_mask
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        opt_input_ids = kwargs.pop("opt_input_ids", None)
        opt_attention_mask = kwargs.pop("opt_attention_mask", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        
        if opt_input_ids is not None:
            _inputs["opt_input_ids"] = opt_input_ids
        
        if opt_attention_mask is not None:
            _inputs["opt_attention_mask"] = opt_attention_mask
            
            
        return _inputs
     


class KnowledgeOPTQformerLlavaLlamaForCausalLM(LlamaForCausalLM, LlavaSAMOPTAttentionKnowledgeMetaForCausalLM):
    config_class = KnowledgeLlavaConfig

    def __init__(self, config):
        super(KnowledgeOPTQformerLlavaLlamaForCausalLM, self).__init__(config)
        self.model = KnowledgeOPTLlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        
        
        qformer_config = InstructBlipQFormerConfig(encoder_hidden_size=config.hidden_size, hidden_size=config.hidden_size,
                                                   vocab_size=config.vocab_size, num_hidden_layers=4,
                                                   num_attention_heads=16)
        self.qformer_query_tokens = nn.Parameter(torch.zeros(1, 64, qformer_config.hidden_size))
        # self.qformer_query_tokens = nn.Parameter(torch.zeros(1, 128, qformer_config.hidden_size))
        # self.qformer_query_tokens = nn.Parameter(torch.zeros(1, 256, qformer_config.hidden_size))
        self.qformer = InstructBlipQFormerModel(qformer_config)
        self.embedding_query = nn.Parameter(torch.zeros(1, config.hidden_size))

        # self.query_tokens = 
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        opt_input_ids: Optional[torch.LongTensor] = None,
        opt_attention_mask: Optional[torch.Tensor] = None,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        sam_images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        bsz = input_ids.size(0)
        
        input_position_ids = input_ids.clone()
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels, 
                knowledge_embeds
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                sam_images,
                opt_input_ids,
                opt_attention_mask
            )
        if (input_position_ids == KNOWLEDGE_QUERY_TOKEN_INDEX).sum() != 0:
            image_embedding_len = inputs_embeds.size(1) - input_position_ids.size(1)
            input_position_ids = torch.concat([torch.ones(bsz, image_embedding_len, device=inputs_embeds.device, dtype=input_position_ids.dtype), 
                                               input_position_ids], dim=1)
            inputs_embeds[input_position_ids == KNOWLEDGE_QUERY_TOKEN_INDEX] += self.embedding_query
        else:
            input_position_ids = None
        
        # qformer
        if knowledge_embeds is not None:
            knowledge_embeds = knowledge_embeds.view(bsz, -1, knowledge_embeds.size(-1))
            print(knowledge_embeds.size())
            knowledge_attention_mask = torch.ones(knowledge_embeds.size()[:-1], dtype=torch.long, device=knowledge_embeds.device)
            query_tokens = self.qformer_query_tokens.expand(knowledge_embeds.shape[0], -1, -1)
            query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=knowledge_embeds.device)
            if qformer_attention_mask is None:
                qformer_attention_mask = torch.ones_like(qformer_input_ids)
            qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
            query_outputs = self.qformer(
                input_ids=qformer_input_ids,
                attention_mask=qformer_attention_mask,
                query_embeds=query_tokens,
                encoder_hidden_states=knowledge_embeds,
                encoder_attention_mask=knowledge_attention_mask,
            )
            query_output = query_outputs[0][:, : query_tokens.size(1), :]
        else:
            query_output = None
        # print(query_output.size())

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_position_ids=input_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            knowledge_embeds=query_output,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        opt_input_ids = kwargs.pop("opt_input_ids", None)
        opt_attention_mask = kwargs.pop("opt_attention_mask", None)
        qformer_input_ids = kwargs.pop("qformer_input_ids", None)
        sam_images = kwargs.pop("sam_images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        
        if opt_input_ids is not None:
            _inputs["opt_input_ids"] = opt_input_ids
        
        if opt_attention_mask is not None:
            _inputs["opt_attention_mask"] = opt_attention_mask
            
        if qformer_input_ids is not None:
            _inputs["qformer_input_ids"] = qformer_input_ids
            
        if sam_images is not None:
            _inputs["sam_images"] = sam_images
            
        return _inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
