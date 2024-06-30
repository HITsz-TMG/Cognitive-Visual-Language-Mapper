import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", type=str, default="")
args = parser.parse_args()

pred_path = args.pred_file

# pred_path = "/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/aokvqa/answers/llava_aokvqa_multichoice_val/llava_7b_attention_opt_knowledge_qa_6_layers/1.75epoch.jsonl"
prediction_data = [json.loads(p) for p in open(pred_path, "r")]

golden_path = "/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/aokvqa/llava_aokvqa_multichoice_val.jsonl"
golden = [json.loads(p) for p in open(golden_path, "r")]

cnt = 0
for idx, pred in enumerate(prediction_data):
    answer = golden[idx]["answer"]
    prediction = pred["text"]
    
    if prediction in ["A", "B", "C", "D"]:
        if answer == prediction:
            cnt += 1
    else:
        print(prediction)

print(cnt / len(prediction_data))