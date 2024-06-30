#!/bin/bash

CKPT="CVLM-LLaVA"
STEP="1epoch"
SPLIT="llava_textvqa_val"

python -m llava.eval.model_vqa_loader \
      --model-base lmsys/vicuna-7b-v1.5 \
      --model-path checkpoints/$CKPT/checkpoint-$STEP \
      --question-file ./playground/knowledge_qa/eval/textvqa/$SPLIT.jsonl \
      --image-folder / \
      --answers-file ./playground/knowledge_qa/eval/textvqa/answers/$SPLIT/$CKPT/$STEP.jsonl \
      --temperature 0 \
      --conv-mode vicuna_v1 &

wait

python llava/eval/eval_textvqa.py --annotation-file ./playground/knowledge_qa/eval/textvqa/TextVQA_0.5.1_val.json --result-file ./playground/knowledge_qa/eval/textvqa/answers/$SPLIT/$CKPT/$STEP.jsonl