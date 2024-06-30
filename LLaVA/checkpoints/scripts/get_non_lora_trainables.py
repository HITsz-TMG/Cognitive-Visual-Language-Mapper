import os.path
import shutil
import torch

ckpt = "llava_vka_qa"
print(ckpt)
step = "2955"
target_dir = "checkpoint-1epoch"

checkpoint_name = f"../{ckpt}/checkpoint-{step}/global_step{step}/mp_rank_00_model_states.pt"
save_name = f"../{ckpt}/{target_dir}/non_lora_trainables.bin"

save_dir = os.path.dirname(save_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

params = torch.load(checkpoint_name, map_location="cpu")
params = params["module"]
train_params = {}
for n, p in params.items():
    if "mm_projector" in n or "knowledge_vision_opt_projector" in n or "qformer_query_tokens" in n or "qformer" in n or "embedding_query" in n:
        train_params[n] = p
        
print(train_params.keys())
torch.save(train_params, save_name)