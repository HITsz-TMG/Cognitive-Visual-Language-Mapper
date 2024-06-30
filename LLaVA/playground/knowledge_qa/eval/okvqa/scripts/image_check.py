import json
import os

with open("/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/okvqa/llava_okvqa_mscoco_val.jsonl", "r") as file:
    lines = file.readlines()
    data = [json.loads(line) for line in lines]
    
for line in data:
    image = line["image"]
    
    image_path = os.path.join("/data/share/datasets/MutilModalDataset/coco_images/train2014", image)
    
    if not os.path.exists(image_path):
        print(image_path)
        image_path = image_path.replace("train", "val")
        
        if not os.path.exists(image_path):
            print(image_path)
    
    line["image"] = image_path

with open("/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/okvqa/llava_okvqa_mscoco_val.jsonl", "w") as file:
    for line in data:
        file.write(json.dumps(line, ensure_ascii=False) + "\n")
        file.flush()
        