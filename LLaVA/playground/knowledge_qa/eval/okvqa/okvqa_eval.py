from VQA.PythonHelperTools.vqaTools.vqa import VQA
from VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
import json

annFile = "VQA/mscoco_val2014_annotations.json"
quesFile = "VQA/OpenEnded_mscoco_val2014_questions.json"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", type=str, default="/data/cxy/LMEye/playground/data/eval/okvqa/answers_upload/llava_okvqa_mscoco_val/lmeye-7b-baseline-1.75epoch-1epoch.json")
args = parser.parse_args()

resFile = args.pred_file
file = json.load(open(resFile, "r"))

vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)

vqaEval = VQAEval(vqa, vqaRes, n=2)

vqaEval.evaluate()

print("\n")
print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
