import os
import json
import copy
import torch
import pathlib
import transformers
from PIL import Image
from torch.utils.data import Dataset
from llava.constants import  DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from transformers import AutoTokenizer
from local_transformers.transformers import CLIPImageProcessor
from decoder_model.model.knowledge_decoder import KnowledgeVisionGPT2Config, KnowledgeVisionGPT2
from decoder_model.model.knowledge_decoder import KnowledgeVisionOPT, KnowledgeVisionOPTConfig
from decoder_model.train.gpt2_trainer import GPT2Trainer
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    lm_model_path: Optional[str] = field(default="facebook/opt-1.3b")
    clip_model_path: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    knowledge_pretrain_path: Optional[str] = field(default=None)
    
@dataclass
class DataArguments:
    data_path: str = field(default="",
                           metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default="/")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="")
    optim: str = field(default="adamw_torch")
    projector_lr: Optional[float] = None
    freeze_clip: Optional[bool] = True
    seed: Optional[int] = 3407
    
def preprocess_plain(
    conversations: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    assert len(conversations) == 2
    # if DEFAULT_IMAGE_TOKEN in conversations[0]["value"]:
    #     conversations[0]["value"] = conversations[0]["value"].replace(DEFAULT_IMAGE_TOKEN, '')
    
    conversations[0]["value"] = conversations[0]["value"].strip()
    conversations[1]["value"] = conversations[1]["value"].strip()
    
    conv = conversations[0]["value"].strip() + conversations[1]["value"] + tokenizer.eos_token
    input_ids = tokenizer(conv, return_tensors="pt").input_ids
    if input_ids.size(0) == 1:
        input_ids = input_ids.squeeze(dim=0)
    targets = copy.deepcopy(input_ids)
    tokenized_len = len(tokenizer(conversations[0]["value"]).input_ids)
    targets[:tokenized_len] = IGNORE_INDEX
    
    return dict(input_ids=input_ids, labels=targets)
    
    
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        
    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i):
        image_file = self.list_data_dict[i]['image']
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        image = processor.preprocess(image, return_tensors='pt', input_data_format="channels_last")['pixel_values'][0]
        
        conversations = self.list_data_dict[i]["conversations"]
        
        data_dict = preprocess_plain(conversations, self.tokenizer)
        # if 'image' in self.list_data_dict[i]:
        data_dict['image'] = image
        # return data_dict
        return {"input_ids": data_dict["input_ids"],
                "labels": data_dict["labels"],
                "images": image}

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    # def __init__(self, tokenizer):
    #     self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                            for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        images = [instance['images'] for instance in instances]
        batch['images'] = torch.stack(images)
        
        return batch

# def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
#                                 data_args) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
#                                 data_path=data_args.data_path,
#                                 data_args=data_args)
#     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
#     return dict(train_dataset=train_dataset,
#                 eval_dataset=None,
#                 data_collator=data_collator)

        
def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    
    config = KnowledgeVisionOPTConfig.from_pretrained(model_args.lm_model_path)
    
    model = KnowledgeVisionOPT.from_pretrained(model_args.lm_model_path, config=config)
    
    if model_args.knowledge_pretrain_path is not None:
        knowledge_weights = torch.load(model_args.knowledge_pretrain_path, map_location="cpu")
        knowledge_weights = {(k[11:] if k.startswith('base_model.') else k): v for k, v in knowledge_weights.items()}
        if any(k.startswith('model.model.') for k in knowledge_weights):
            knowledge_weights = {(k[6:] if k.startswith('model.') else k): v for k, v in knowledge_weights.items()}
        model.load_state_dict(knowledge_weights, strict=False)
    
    # 视觉模型训练参数关闭
    if training_args.freeze_clip:
        model.clip_model.requires_grad_(False)
        
    tokenizer = AutoTokenizer.from_pretrained(model_args.lm_model_path, add_bos_token=True)
    # tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.model_max_length = 2048
    image_processor = CLIPImageProcessor.from_pretrained(model_args.clip_model_path)
    data_args.image_processor = image_processor
    
    trainer = GPT2Trainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    train_dataset=LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args),
                    data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer))
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    
    model.config.use_cache = True
    
    
if __name__ == "__main__":
    train()