import json
import argparse
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator

parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", type=str,
                    default="/data/cxy/models/Qwen-VL/data/aokvqa/answers/QWen_VL_VKA_2epoch_aokvqa_240406045328_fs0_s0.json")
args = parser.parse_args()

pred_path = args.pred_file

# pred_path = "/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/aokvqa/answers/llava_aokvqa_openended_val/llava_7b_attention_opt_knowledge_qa_6_layers/1.75epoch.jsonl"
prediction_data = [json.loads(p) for p in open(pred_path, "r")]
# prediction_data = json.load(open(pred_path, "r"))

golden_path = "/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/aokvqa/llava_aokvqa_openended_val.jsonl"
golden = [json.loads(p) for p in open(golden_path, "r")]

cnt = 0
pred_list = []
for idx, pred in enumerate(prediction_data):
    answers = golden[idx]["answer"]
    # prediction = pred["answer"]
    prediction = pred["text"]
    pred_list.append({
        "pred_answer": prediction,
        "gt_answers": answers,
    })
    
    for answer in answers:
        if prediction.lower() == answer.lower():
            cnt += 1
            break

print(cnt / len(prediction_data))

evaluator = TextVQAAccuracyEvaluator()
print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))