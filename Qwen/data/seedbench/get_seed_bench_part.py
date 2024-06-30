import json

# data = json.load(open("/data/cxy/models/Qwen-VL/data/seedbench/qwen-seed-bench_answer.jsonl", "r"))

data = [json.loads(q) for q in open("/data/cxy/models/Qwen-VL/data/seedbench/qwen-seed-bench_answer.jsonl")]
print(len(data))
results = []
for line in data:
    if "png" not in line["image"]:
        results.append(line)
print(len(results))
with open("/data/cxy/models/Qwen-VL/data/seedbench/qwen-seed-bench-image_answer.jsonl", "w") as file:
    for line in results:
        file.write(json.dumps(line, ensure_ascii=False) + "\n")
        file.flush()
# json.dump(data, )