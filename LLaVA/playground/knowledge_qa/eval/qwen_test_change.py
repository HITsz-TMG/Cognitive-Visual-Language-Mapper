# 将测试数据集转化为Qwen-VL的格式

import json
import os
base_path = "/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval"
file_name = "textvqa/llava_textvqa_val.jsonl"
task = "vqa"

target_dir = "/data/cxy/models/Qwen-VL/data"
target_name = file_name.replace("llava", "qwen")
target_path = os.path.join(target_dir, target_name)

dir = os.path.dirname(target_path)
if not os.path.exists(dir):
    os.makedirs(dir)

file_path = os.path.join(base_path, file_name)

data = [json.loads(q) for q in open(file_path, "r")]
mapping2index = {"A":0, "B":1, "C":2, "D":3, "E":4}

results = []
for line in data:
    if task == "vqa":
        question = line["text"].split("\n")[0]
        
        tmp = {
            "question": question,
            "image": os.path.join("/data/share/datasets/MutilModalDataset/open_images", line["image"]),
            "question_id": line["question_id"]
        }
        results.append(tmp)
    else:
        question = line["text"].split("\n")[0]
        choices = line["text"].split("\n")[1:5]
        
        choices = [choice[3:] for choice in choices]
        
        
        tmp = {
            "question": question,
            "choices": choices,
            "answer": mapping2index[line["answer"]],
            "image": line["image"],
            "category": line["category"]
        }
        
        results.append(tmp)

with open(target_path, "w") as file:
    for line in results:
        file.write(json.dumps(line, ensure_ascii=False) + "\n")
        file.flush()
    

