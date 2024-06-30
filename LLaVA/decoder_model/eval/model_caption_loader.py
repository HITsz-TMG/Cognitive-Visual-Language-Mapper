import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import json
from tqdm import tqdm
import shortuuid
from decoder_model.model.builder import load_pretrained_model
from torch.utils.data import Dataset, DataLoader
from llava.utils import disable_torch_init

from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        # qs = tokenizer.bos_token
        prompt = "Give the background knowledge relevant to the image."
        
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.squeeze(dim=0)

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)

# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

    
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, image_processor = load_pretrained_model(model_path, args.model_base)
    model = model.to(device='cuda', non_blocking=True)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    # data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        
        with torch.inference_mode():
            # output_ids = model.generate(
            #     input_ids,
            #     images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True))
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                no_repeat_ngram_size=3,
                use_cache=True)
        
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": "test",
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/cxy/Knowledge_LLaVA/checkpoints/opt-knowledge-caption_pretrain_2M_prompt_continue_merge/checkpoint-89320/pytorch_model.bin")
    parser.add_argument("--model-base", type=str, default="/data/share/Model/opt-1.3b")
    parser.add_argument("--image-folder", type=str, default="/")
    parser.add_argument("--question-file", type=str, default="/data/cxy/Knowledge_LLaVA/playground/knowledge_data/eval/wikipedia/wikipedia_test_5k.jsonl")
    parser.add_argument("--answers-file", type=str, default="/data/cxy/Knowledge_LLaVA/playground/knowledge_data/eval/wikipedia/answers/answer_5k.jsonl")
    parser.add_argument("--conv-mode", type=str, default="plain")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2000)
    args = parser.parse_args()

    eval_model(args)