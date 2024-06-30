import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", type=str, default="/data/cxy/LMEye/playground/data/eval/seedbench/answers/llava-seed-bench_answer/lmeye-7b-baseline-1.5epoch/1epoch.jsonl")
args = parser.parse_args()

pred_path = args.pred_file

# pred_path = "/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/aokvqa/answers/llava_aokvqa_multichoice_val/llava_7b_attention_opt_knowledge_qa_6_layers/1.75epoch.jsonl"
prediction_data = [json.loads(p) for p in open(pred_path, "r")]

golden_path = "playground/knowledge_qa/eval/seedbench/llava-seed-bench_answer.jsonl"
golden = [json.loads(p) for p in open(golden_path, "r")]

cnt = 0
cate2acc = {"Scene Understanding": 0, "Instance Identity": 0, "Instance Attributes": 0, "Instance Location": 0, 
            "Instances Counting": 0, "Spatial Relation": 0, "Instance Interaction": 0, "Visual Reasoning": 0,
            "Text Understanding": 0, "Action Recognition": 0, "Action Prediction": 0, "Procedure Understanding":0}
cate2total = {"Scene Understanding": 0, "Instance Identity": 0, "Instance Attributes": 0, "Instance Location": 0, 
            "Instances Counting": 0, "Spatial Relation": 0, "Instance Interaction": 0, "Visual Reasoning": 0,
            "Text Understanding": 0, "Action Recognition": 0, "Action Prediction": 0, "Procedure Understanding":0}

cnt_spatial = 0
acc_spatial = 0
cnt_temporal = 0
acc_temporal = 0

tt = 0
for idx, pred in enumerate(prediction_data):
    answer = golden[idx]["answer"]
    category = golden[idx]["category"]
    prediction = pred["text"]
    
    if category == "Text Understanding":
        if "color" in golden[idx]["text"]:
            tt += 1
    
    if prediction in ["A", "B", "C", "D"]:
        if answer == prediction:
            cnt += 1
            cate2acc[category] += 1
            if category == "Action Recognition" or category == "Action Prediction" or category == "Procedure Understanding":
                acc_temporal += 1
            else:
                acc_spatial += 1
        
    else:
        print(prediction)
        
    if category == "Action Recognition" or category == "Action Prediction" or category == "Procedure Understanding":
        cnt_temporal += 1
    else:
        cnt_spatial += 1
    
    
    cate2total[category] += 1
print(tt)
print(cnt_spatial)
print(acc_spatial / cnt_spatial)
print(acc_temporal / cnt_temporal)
print(cnt / len(prediction_data))

for cate, acc in cate2acc.items():
    print(f"Category: {cate}  Acc: {acc / cate2total[cate]} Len: {cate2total[cate]}")