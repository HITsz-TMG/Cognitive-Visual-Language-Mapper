#!/bin/bash

CKPT="CVLM-LLaVA"
STEP="1epoch"
SPLIT="llava-seed-bench"

python -m llava.eval.model_vqa_loader \
      --model-base lmsys/vicuna-7b-v1.5 \
      --model-path checkpoints/$CKPT/checkpoint-$STEP \
      --question-file ./playground/knowledge_qa/eval/seedbench/$SPLIT.jsonl \
      --image-folder / \
      --answers-file ./playground/knowledge_qa/eval/seedbench/answers/$SPLIT/$CKPT/$STEP.jsonl \
      --temperature 0 \
      --conv-mode vicuna_v1 &

wait

python playground/knowledge_qa/eval/seedbench/seed_bench_eval.py --pred_file ./playground/knowledge_qa/eval/seedbench/answers/$SPLIT/$CKPT/$STEP.jsonl
