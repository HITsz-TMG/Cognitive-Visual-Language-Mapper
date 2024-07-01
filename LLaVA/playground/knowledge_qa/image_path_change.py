import json

path = "LLaVA\playground\knowledge_qa\eval\aokvqa\llava_aokvqa_multichoice_val.jsonl"

data = [json.loads(q) for q in open(path, "r")]

for line in data:
    image = line["image"]
    
    if "coco_images" in image:
        line["image"] = "coco_images" + image.split("coco_images")[1]
        
    elif "infoseek_val_images" in image:
        line["image"] = "infoseek_val_images" + image.split("infoseek_val_images")[1]
        
with open(path, "w") as file:
    for line in data:
        file.write(json.dumps(line, ensure_ascii=False) + "\n")
        file.flush()