#!/bin/bash

CKPT="CVLM-LLaVA"
STEP="1epoch"
SPLIT="llava_aokvqa_multichoice_val"

python -m llava.eval.model_vqa_loader \
      --model-base lmsys/vicuna-7b-v1.5 \
      --model-path checkpoints/$CKPT/checkpoint-$STEP \
      --question-file ./playground/knowledge_qa/eval/aokvqa/$SPLIT.jsonl \
      --image-folder / \
      --answers-file ./playground/knowledge_qa/eval/aokvqa/answers/$SPLIT/$CKPT/$STEP.jsonl \
      --temperature 0 \
      --conv-mode vicuna_v1 &

wait

python playground/knowledge_qa/eval/aokvqa/aokvqa_multichoice_eval.py --pred_file ./playground/knowledge_qa/eval/aokvqa/answers/$SPLIT/$CKPT/$STEP.jsonl