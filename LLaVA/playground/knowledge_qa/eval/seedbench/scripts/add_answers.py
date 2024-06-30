import json

data = json.load(open("/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/seedbench/SEED-Bench.json", "r"))["questions"]
new_data = []
for line in data:
    if line["question_id"] != "v4967":
        new_data.append(line)
        
p_data = [json.loads(p) for p in open("/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/seedbench/llava-seed-bench.jsonl", "r")]


for idx, line in enumerate(p_data):
    line["answer"] = new_data[idx]["answer"]
    
with open("/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/seedbench/llava-seed-bench_answer.jsonl", "w") as file:
    for line in p_data:
        file.write(json.dumps(line, ensure_ascii=False) + "\n")
        file.flush()