# 评测Knowledge预训练完成的模型
import argparse
from Qwen_VL.configuration_qwen import QWenConfig
from Qwen_VL.modeling_qwen import QWenLMHeadModel
from Qwen_VL.tokenization_qwen import QWenTokenizer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/data/cxy/models/Qwen-VL/checkpoints/output_qwen/checkpoint-31552')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()
    
    model = QWenLMHeadModel.from_pretrained(args.checkpoint).eval()
    
    model = model.cuda()
    
    tokenizer = QWenTokenizer.from_pretrained(args.checkpoint)
    
    prompt = "<img>Q229/Q22932727.jpg</img>\nGive the background knowledge relevant to the image."
    
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids
    
    pred = model.generate(
        input_ids=prompt_tokens.cuda(),
        do_sample=False,
        num_beams=1,
        max_new_tokens=100
    )
    answers = [
    tokenizer.decode(_[prompt_tokens.size(1):].cpu(),
                        skip_special_tokens=True).strip() for _ in pred
    ]
    print(answers)
    
    