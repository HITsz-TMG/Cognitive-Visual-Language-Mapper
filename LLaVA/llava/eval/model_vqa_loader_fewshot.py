import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import json
import time
from tqdm import tqdm
import shortuuid
import transformers

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_KNOWLEDGE_QUERY_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

data2fewshot = []


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, few_shot_data, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.few_shot_data = few_shot_data
        self.image_folder = image_folder
        self.opt_tokenizer = transformers.AutoTokenizer.from_pretrained("/data/share/Model/opt-1.3b", add_bos_token=True)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        for i in range(0, 4):
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + self.few_shot_data[i]["question"])
            conv.append_message(conv.roles[1], self.few_shot_data[i]["answer"])
        
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(prompt)

        few_shot_images = [Image.open(t["image"]).convert('RGB') for t in self.few_shot_data]
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        few_shot_images.append(image)
        image_tensor = process_images(few_shot_images, self.image_processor, self.model_config)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        opt_prompt = "Give the background knowledge relevant to the image."
        opt_input_ids = self.opt_tokenizer(opt_prompt, return_tensors="pt").input_ids.squeeze(dim=0)
        opt_attention_mask = opt_input_ids.ne(self.opt_tokenizer.pad_token_id)

        return input_ids, image_tensor, opt_input_ids, opt_attention_mask

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, few_shot_data, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, few_shot_data, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


task2_few_shot = {
    "okvqa": "/data/cxy/models/Qwen-VL/data/few_shot_data/okvqa_few_shot.jsonl",
    "infoseek": "/data/cxy/models/Qwen-VL/data/few_shot_data/infoseek_few_shot.jsonl",
    "aokvqa_r": "/data/cxy/models/Qwen-VL/data/few_shot_data/aokvqa_few_shot_rationale_r1.jsonl"
}

def eval_model(args):
    # Model

    global qformer_tokenizer
    qformer_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/data/cxy/Knowledge_LLaVA/local_models/qformer_tokenizer")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = "llava-" + get_model_name_from_path(model_path) + "-lora"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    few_shot_data = [json.loads(q) for q in open(task2_few_shot["aokvqa_r"], "r")]
    # questions = questions[:100]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, few_shot_data, args.image_folder, tokenizer, image_processor, model.config)

    total_time = 0
    for (input_ids, image_tensor, opt_input_ids, opt_attention_mask), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        opt_input_ids = opt_input_ids.to(device='cuda', non_blocking=True)
        opt_attention_mask = opt_attention_mask.to(device='cuda', non_blocking=True)

        
        start_time = time.time()
        if "opt_knowledge" in model_path:
            print("opt_knowledge....")
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    opt_input_ids=opt_input_ids,
                    opt_attention_mask=opt_attention_mask,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)
        else:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)
                
        end_time = time.time()
        used_time = end_time - start_time
        total_time += used_time

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()
    print(total_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/cxy/Knowledge_LLaVA/checkpoints/llava_7b_attention_opt_knowledge_qa_665K_2epoch_pretrain/checkpoint-1epoch")
    parser.add_argument("--model-base", type=str, default="/data/share/Model/vicuna-7b-v1.5")
    parser.add_argument("--image-folder", type=str, default="/")
    parser.add_argument("--question-file", type=str, default="./playground/knowledge_qa/eval/okvqa/llava_okvqa_mscoco_val.jsonl")
    parser.add_argument("--answers-file", type=str, default="./playground/knowledge_qa/eval/okvqa/answers/llava_okvqa_mscoco_val/llava_7b_knowledge_qa_baseline_r/llava_7b_knowledge_qa_baseline_r.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--sam_image_folder", type=str, default="/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/sam/images")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
