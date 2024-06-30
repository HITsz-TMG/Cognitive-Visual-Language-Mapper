# 仅获取Knowledge Benchmark的评测数据
import json

golden_data = [json.loads(q) for q in open("/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/okvqa/llava_okvqa_mscoco_val.jsonl", "r")]

okvqa_category = json.load(open("/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/okvqa/okvqa_category.json", "r"))
print(len(okvqa_category))
results = []
for line in golden_data:
    question_id = str(line["question_id"])
    if question_id not in okvqa_category:
        continue
    
    line["category"] = okvqa_category[question_id][0]
    line["golden"] = okvqa_category[question_id][1]
    
    line["text"] = line["text"].split("\n")[0]
    
    results.append(line)
    
with open("/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/okvqa/okvqa_val_category.jsonl", "w") as file:
    for line in results:
        file.write(json.dumps(line, ensure_ascii=False) + "\n")
        file.flush()