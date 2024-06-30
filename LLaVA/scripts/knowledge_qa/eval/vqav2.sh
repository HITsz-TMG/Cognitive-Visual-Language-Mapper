#!/bin/bash

CKPT="CVLM-LLaVA"
STEP="1epoch"
SPLIT="llava_vqav2_mscoco_val"

python -m llava.eval.model_vqa_loader \
      --model-base lmsys/vicuna-7b-v1.5 \
      --model-path checkpoints/$CKPT/checkpoint-$STEP \
      --question-file ./playground/knowledge_qa/eval/vqav2/$SPLIT.jsonl \
      --image-folder / \
      --answers-file ./playground/knowledge_qa/eval/vqav2/answers/$SPLIT/$CKPT/$STEP.jsonl \
      --temperature 0 \
      --conv-mode vicuna_v1 &

wait

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT --step $STEP