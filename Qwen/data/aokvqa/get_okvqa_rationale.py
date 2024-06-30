import json

data = [json.loads(q) for q in open("/data/cxy/models/Qwen-VL/data/aokvqa/qwen_aokvqa_openended_val.jsonl", "r")]


for line in data:
    line["question"] = "Answer the following question with a word or phrase and generate the rationale with one sentence. Question: " + line["question"]
    
with open("/data/cxy/models/Qwen-VL/data/aokvqa/qwen_aokvqa_openended_val_rationale.jsonl", "w") as file:
    for line in data:
        file.write(json.dumps(line, ensure_ascii=False) + "\n")
        file.flush()