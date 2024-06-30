# 直接使用生成的方法来评测
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12357'
import argparse
import itertools
import random
import json
import time
from functools import partial

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from Qwen_VL.modeling_qwen import QWenLMHeadModel
from Qwen_VL.tokenization_qwen import QWenTokenizer

multiple_choices = ['A', 'B', 'C', 'D', 'E']

ds_collections = {
    'scienceqa_test_img': {
        'test': 'data/scienceqa/scienceqa_test_img.jsonl',
    },
    'aokvqa': {
        'test': 'data/aokvqa/qwen_aokvqa_multichoice_val.jsonl',
    },
    'seedbench': {
        'test': 'data/seedbench/qwen-seed-bench-image_answer.jsonl'
    }
}


# def collate_fn(batches, pad_token_id):

#     input_tokens = [_['input_tokens'] for _ in batches]
#     target_lengths = [_['target_lengths'] for _ in batches]
#     answers = [_['answer'] for _ in batches]

#     chunk_sizes = [len(_) for _ in input_tokens]

#     input_tokens = [_ for _ in itertools.chain.from_iterable(input_tokens)]

#     max_lengths = max([len(_) for _ in input_tokens])
#     input_tokens = [[pad_token_id] * (max_lengths - len(_)) + _
#                     for _ in input_tokens]
#     input_tokens = torch.LongTensor(input_tokens)

#     attention_mask = 1 - input_tokens.eq(pad_token_id).float()

#     return input_tokens, attention_mask, target_lengths, answers, chunk_sizes
def collate_fn(batches, tokenizer):

    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]

    input_ids = tokenizer(questions, return_tensors='pt', padding='longest')

    return question_ids, input_ids.input_ids, input_ids.attention_mask, annotations


class MultipleChoiceDataste(torch.utils.data.Dataset):

    def __init__(self, test, prompt, tokenizer):
        self.datas = open(test).readlines()
        self.prompt = prompt
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):

        data = json.loads(self.datas[idx].strip())
        image = data['image']
        # hint = data['hint'] if data['hint'] else 'N/A'
        hint = 'N/A'
        question = data['question']

        choices = data['choices']
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append('{}. {}'.format(multiple_choices[i], c))
        choice_txt = '\n'.join(choice_list)

        prompt = self.prompt.format(image, question, choice_txt)
        # print(prompt)
        # prompt_tokens = self.tokenizer(prompt).input_ids

        return {
            'question': prompt,
            "question_id": image,
            'annotation': data["answer"]
        }
                
        # return {
        #     'input_tokens': [prompt_tokens + _ for _ in target_tokens],
        #     'target_lengths': [len(_) for _ in target_tokens],
        #     'answer': data['answer'],
        # }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/qwen-pretrain')
    parser.add_argument('--adapter', type=str, default="checkpoints/qwen-vka-stage2")
    parser.add_argument('--dataset', type=str, default='aokvqa')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model = QWenLMHeadModel.from_pretrained(args.checkpoint, device_map='cuda').eval()

    model_path = args.adapter
    if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    
    from peft import PeftModel
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')

    tokenizer = QWenTokenizer.from_pretrained(args.checkpoint)
    
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id

    # prompt = '<img>{}</img>Context: {}\nQuestion: {}\nOptions: {}\nAnswer:'
    prompt = '<img>{}</img>Question: {}\nOptions: {}\nAnswer:'
    
    random.seed(args.seed)
    dataset = MultipleChoiceDataste(test=ds_collections[args.dataset]['test'],
                                    prompt=prompt,
                                    tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    cnt = 0
    total = 0
    outputs = []
    mapping = {0: "A", 1:"B", 2:"C", 3:"D"}
    for _, (question_ids, input_ids, attention_mask,
            annotations) in tqdm(enumerate(dataloader)):
        # with torch.no_grad():
        pred = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=10,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
        )
        answers = [
            tokenizer.decode(_[input_ids.size(1):].cpu(),
                            skip_special_tokens=True).strip() for _ in pred
        ]
        print(answers)
        for question_id, answer, annotation in zip(question_ids, answers,
                                                   annotations):
            outputs.append({
                "image": question_id,
                'answer': answer,
            })
        
        for idx in range(len(answers)):
            golden = mapping[annotations[idx]]
            
            if golden == answers[idx][0].upper():
                cnt += 1
            total += 1
            
    print(cnt / total)
    
    output_path = os.path.join("data/", args.dataset, "answers"
                               )
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'QWen_VL_VKA_1epoch_{args.dataset}_{time_prefix}_s{args.seed}.json'
        results_file = os.path.join(output_path, results_file)
        json.dump(outputs, open(results_file, 'w'), ensure_ascii=False)
    # with torch.no_grad():
    #     for _, (input_tokens, attention_mask, target_lengths, answer,
    #             chunk_sizes) in tqdm(enumerate(dataloader)):

    #         outputs = model(
    #             input_ids=input_tokens[:, :-1].cuda(),
    #             attention_mask=attention_mask[:, :-1].cuda(),
    #             return_dict=True,
    #         )
    #         losses = torch.nn.functional.cross_entropy(outputs.logits.permute(
    #             0, 2, 1),
    #                                                    input_tokens[:,
    #                                                                 1:].cuda(),
    #                                                    reduction='none')

    #         losses = losses.split(chunk_sizes, dim=0)

    #         for loss, target_length, answer in zip(losses, target_lengths,
    #                                                answer):

    #             target_loss = loss.mean(-1)
    #             for _ in range(len(target_length)):
    #                 target_loss[_] = loss[_, -target_length[_]:].mean()
    #             pred = target_loss.argmin().item()
    #             if pred == answer:
    #                 results.append(1)
    #             else:
    #                 results.append(0)

    # torch.distributed.barrier()

    # world_size = torch.distributed.get_world_size()
    # merged_results = [None for _ in range(world_size)]
    # torch.distributed.all_gather_object(merged_results, results)

    # merged_results = [_ for _ in itertools.chain.from_iterable(merged_results)]

    # if torch.distributed.get_rank() == 0:
    #     print(f"Evaluating {args.dataset} ...")
    #     print(f'Acc@1: {sum(merged_results) / len(merged_results)}')

    # torch.distributed.barrier()
