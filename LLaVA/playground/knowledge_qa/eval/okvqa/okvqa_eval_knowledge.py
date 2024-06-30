# 细粒度评测
import json
CKPT="llava_7b_attention_opt_knowledge_qa_sam_transformers_qformer_ptuning8_stage2"
STEP="1epoch"

# predict_file = f"/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/okvqa/answers_upload/llava_okvqa_mscoco_val/{CKPT}-{STEP}.json"

predict_file = "/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/okvqa/answers_upload/llava_okvqa_mscoco_val/llava_7b_attention_opt_knowledge_qa_665K-2epoch.json"
predict_data = json.load(open(predict_file, "r"))

golden_data = [json.loads(q) for q in open("/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/okvqa/llava_okvqa_mscoco_val.jsonl", "r")]

okvqa_category = json.load(open("/data/cxy/Knowledge_LLaVA/playground/knowledge_qa/eval/okvqa/okvqa_category.json", "r"))

acc = 0
total = 0
cate2total = {"Vehicles and Transportation":0, "Brands, Companies and Products":0, "Objects, Material and Clothing":0,
              "Sports and Recreation":0, "Cooking and Food":0, "Geography, History, Language and Culture":0,
              "People and Everyday life":0, "Plants and Animals":0, "Science and Technology":0,
              "Weather and Climate":0, "Other":0}
cate2acc = {"Vehicles and Transportation":0, "Brands, Companies and Products":0, "Objects, Material and Clothing":0,
              "Sports and Recreation":0, "Cooking and Food":0, "Geography, History, Language and Culture":0,
              "People and Everyday life":0, "Plants and Animals":0, "Science and Technology":0,
              "Weather and Climate":0, "Other":0}
for line in predict_data:
    pred = line["answer"]
    question_id = str(line["question_id"])
    if question_id not in okvqa_category:
        continue
    
    category = okvqa_category[question_id][0]
    answers = okvqa_category[question_id][1]    
    total += 1
    if category not in cate2total:
        cate2total[category] = 1
    else:
        cate2total[category] += 1
        
    for answer in answers:
        ans = answer["answer"]
        # if ans.lower() == pred.lower():
        if ans.lower() in pred.lower():
        # if pred.lower() in ans.lower():
            acc += 1
            if category not in cate2acc:
                cate2acc[category] = 1
            else:
                cate2acc[category] += 1
            break
        
print(acc/total)
for cate, acc in cate2acc.items():
    print(f"Category: {cate:<50} Acc:{acc / cate2total[cate] * 100:<20} Len:{cate2total[cate]}")
    
    

