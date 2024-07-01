import json
import re
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str, default="/data/cxy/models/Qwen-VL/data/infoseek/answers/QWen_VL_tuned_infoseek_240406023717_fs0_s0.json")
args = parser.parse_args()

pred_path = args.pred_file
# pred_path = "/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/infoseek/answers/llava_infoseek_val/llava_7b_attention_opt_knowledge_qa/1epoch.jsonl"
# pred_path = "/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/infoseek/answers/llava_infoseek_val/llava_7b_knowledge_qa_baseline_r/llava_7b_knowledge_qa_baseline_r.jsonl"
prediction_data = [json.loads(p) for p in open(pred_path, "r")]
# prediction_data = json.load(open(pred_path))

golden_path = "/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/infoseek/llava_infoseek_val.jsonl"
golden = [json.loads(p) for p in open(golden_path, "r")]

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

def extract_number(s):
    # 使用正则表达式匹配包含数字的部分
    match = re.search(r'\d+(\.\d+)?', s)
    
    # 如果匹配成功，返回匹配的部分作为浮点数
    if match:
        return float(match.group())
    else:
        return None

cnt = 0
cate2total = {}
cate2acc = {}
for idx, pred in enumerate(prediction_data):
    answers = golden[idx]["answer"]
    category = golden[idx]["category"]
    if category not in cate2total:
        cate2total[category] = 1
    else:
        cate2total[category] += 1
    
    prediction = pred["text"]
    
    if isinstance(answers[0], str):
        for answer in answers:
            # if prediction.lower() == answer.lower():
            # if prediction.lower() in answer.lower():
            if answer.lower() in prediction.lower():
                # print(f"Question: {golden[idx]['text']} Prediction: {prediction}")
                if category not in cate2acc:
                    cate2acc[category] = 1
                else:
                    cate2acc[category] += 1
                cnt += 1
                break
    else:
        range_answer = answers[0]["range"]
        if is_float(prediction):
            if float(prediction) >= range_answer[0] and float(prediction) <= range_answer[1]:
                if category not in cate2acc:
                    cate2acc[category] = 1
                else:
                    cate2acc[category] += 1
                cnt += 1
        else:
            prediction = extract_number(prediction)
            if prediction is not None:
                if float(prediction) >= range_answer[0] and float(prediction) <= range_answer[1]:
                    if category not in cate2acc:
                        cate2acc[category] = 1
                    else:
                        cate2acc[category] += 1
                    cnt += 1

print(cnt)
print(cnt / len(prediction_data))

for cate, acc in cate2acc.items():
    print(f"Category: {cate} Acc:{acc / cate2total[cate]}")
        