#!/bin/bash

CKPT="CVLM-LLaVA"
STEP="1epoch"
SPLIT="llava_infoseek_val"

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
      --model-base lmsys/vicuna-7b-v1.5 \
      --model-path checkpoints/$CKPT/checkpoint-$STEP \
      --question-file ./playground/knowledge_qa/eval/infoseek/$SPLIT.jsonl \
      --image-folder / \
      --answers-file ./playground/knowledge_qa/eval/infoseek/answers/$SPLIT/$CKPT/$STEP.jsonl \
      --temperature 0 \
      --conv-mode vicuna_v1 &

wait
      
python playground/knowledge_qa/eval/infoseek/infoseek_eval.py --pred_file ./playground/knowledge_qa/eval/infoseek/answers/$SPLIT/$CKPT/$STEP.jsonl