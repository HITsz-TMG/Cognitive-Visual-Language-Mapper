#!/bin/bash

CKPT="CVLM-LLaVA"
STEP="1epoch"
SPLIT="llava_okvqa_mscoco_val"

python -m llava.eval.model_vqa_loader \
      --model-base lmsys/vicuna-7b-v1.5 \
      --model-path checkpoints/$CKPT/checkpoint-$STEP \
      --question-file ./playground/knowledge_qa/eval/okvqa/$SPLIT.jsonl \
      --image-folder / \
      --answers-file ./playground/knowledge_qa/eval/okvqa/answers/$SPLIT/$CKPT/$STEP.jsonl \
      --temperature 0 \
      --conv-mode vicuna_v1 &

wait

python scripts/convert_okvqa_for_submission.py --split $SPLIT --ckpt $CKPT --step $STEP