import os.path
import shutil

import torch

ckpt = "llava_opt_pretrain"
print(ckpt)
step = "10695"

checkpoint_name = f"../{ckpt}/checkpoint-{step}/global_step{step}/mp_rank_00_model_states.pt"
save_name = f"../{ckpt}/train_params_{step}.bin"

params = torch.load(checkpoint_name, map_location="cpu")
params = params["module"]
unforzen_params = {}
flag = 0
for n, p in params.items():
    if "knowledge_vision_opt_projector" in n:
        print(n)
        unforzen_params[n] = p

torch.save(unforzen_params, save_name)


