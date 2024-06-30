from VQA.PythonHelperTools.vqaTools.vqa import VQA
from VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
import json

annFile = "VQA/val_annotations_10000_no_shuffule.json"
quesFile = "VQA/val_question_10000_no_shuffule.json"

resFile = "/data/cxy/LMEye/playground/data/eval/vqav2/answers_upload/llava_vqav2_mscoco_val/lmeye-7b-baseline-1.5epoch-1epoch.json"
file = json.load(open(resFile, "r"))
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)

vqaEval = VQAEval(vqa, vqaRes, n=2)

vqaEval.evaluate()

print("\n")
print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))