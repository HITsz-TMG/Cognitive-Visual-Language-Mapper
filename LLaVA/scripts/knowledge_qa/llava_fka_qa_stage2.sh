#!/bin/bash
export PYTHONPATH=/data/cxy/Knowledge_LLaVA
deepspeed --include localhost:0,1,2,3 --master_port 60002 llava/train/train_mem_knowledge.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 3e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5/ \
    --version v1 \
    --data_path playground/knowledge_qa/LLaVA_KnowledgeQA_504K_pquery.json \
    --pretrain_mm_mlp_adapter llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --image_folder path_to_store_image \
    --vision_tower openai/local_models/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava_fka_qa_stage2 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.125 \
    --save_total_limit 3 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --gradient_checkpointing True \
    --knowledge_sam_stage2 True \
    --lora_load True \
    --pretrain_knowledge_params_path ./checkpoints/llava_fka_qa/checkpoint-1epoch/non_lora_trainables.bin \
    
