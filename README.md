
# Cognitive Visual-Language Mapper: Advancing Multimodal Comprehension with Enhanced Visual Knowledge Alignment

<font size=2><div align='center' >  [[📖 arXiv Paper](https://arxiv.org/abs/2402.13561)] [[📊 Dataset ](https://huggingface.co/datasets/Ghaser/Wikipedia-Knowledge-2M)] </div></font>

## 🌟 Dataset

We released two millions Wikipedia Knowledge Dataset in [Wikipedia-Knowledge-2M](https://huggingface.co/datasets/Ghaser/Wikipedia-Knowledge-2M). The dataset includes a json file and a compressed archive containing all the image files. The image attributes in the JSON file correspond one-to-one with the image files in the compressed archive.

If you need to use our training code below, you can place the JSON file in the LLaVA/playground/knowledge_data directory.

We have also provided the JSON file for the 504K [KonwledgeQA dataset.](https://huggingface.co/datasets/Ghaser/LLaVA-KnowledgeQA-504K). The images in this dataset come from COCO Caption and TextVQA, which you will need to download yourself.


## Environment
- Pytorch `2.0.1`
```shell
pip install -r requirement.txt
```
## Train
### Pretraining Visual Knowledge Aligner

After you have successfully downloaded the Wikipedia files and placed them in the appropriate path, you can use the following code to perform VKA pretraining.

```shell
cd LLaVA
export PYTHONPATH=path_to_current_dir
bash scripts/decoder_model/pretrain_knowledge.sh
```


### LLaVA

#### Training Visual Knowledge Aligner with LLMs

``` shell
bash scripts/knowledge/pretrain.sh
```

You can use the [code](LLaVA/checkpoints/scripts/get_train_checkpoints.py) to extract trainable parameters from the saved checkpoints file and store them for use as input in the next stage of training.

#### Fine-tune VKA on the Question Answering task

``` shell
bash scripts/knowledge_qa/llava_vka_qa.sh
```

Change the `pretrain_knowledge_params_path` to the path where the parameters extracted in the previous stage are stored.

Besides, after completing the training, you can use the [code](LLaVA/checkpoints/scripts/get_non_lora_trainables.py) to extract both trainable non-LoRA parameters and LoRA parameters from the checkpoints.

#### Fine-tune FKA on the Question Answering task.

Finally, we used a two-stage training method when fine-tuning FKA.

``` shell
bash scripts/knowledge_qa/llava_fka_qa.sh
```

``` shell
bash scripts/knowledge_qa/llava_fka_qa_stage2.sh
```

It is important to note that during each stage of training, the parameters from the previous stage need to be accessed via the `pretrain_knowledge_params_path`. And the parameters should be extraxted by [code](LLaVA/checkpoints/scripts/get_non_lora_trainables.py).

### Qwen-VL

#### Training Visual Knowledge Aligner with LLMs

This stage of training also requires loading the training parameters from the Pretraining Visual Knowledge Aligner.
You need to modify attribute `pretrain_opt_adapter` by your save path.
```shell
cd Qwen
bash finetune/pretrain_ds.sh
```

#### Fine-tune VKA on the Question Answering task

```shell
bash finetune/finetune_lora_ds.sh
```

## Evaluation

We released the best model based on LLaVA on [CVLM-LLaVA](https://huggingface.co/Ghaser/CVLM-LLaVA) and best model based on QWen-VL on [CVLM-Qwen](https://huggingface.co/Ghaser/CVLM-Qwen)

### LLaVA

The evaluation scripts of LLaVA are on `scripts/knowledge_qa/eval`,

We mainly evaluated on six benchmark datasets: OK-VQA, VQAv2, A-OKVQA, TextVQA, InfoSeek, and SEED-Bench.

#### OK-VQA
It is important to note that the saved result files will be in the answers_upload folder within the corresponding directory.
```shell
bash scripts/knowledge_qa/eval/okvqa.sh
cd /data/cxy/Knowledge_LLaVA/upload/playground/knowledge_qa/eval/okvqa
python okvqa_eval.py --pred_file your_save_path
```

#### VQAv2

```shell
bash scripts/knowledge_qa/eval/vqav2.sh
cd /data/cxy/Knowledge_LLaVA/upload/playground/knowledge_qa/eval/vqav2
python vqa_eval.py --pred_file your_save_path
```

#### A-OKVQA

Evaluation on open-ended A-OKVQA. The following scripts will also perform the evaluation.

```shell
bash scripts/knowledge_qa/eval/aokvqa_oe.sh
```

Evaluation on multi-choices A-OKVQA

```shell
bash scripts/knowledge_qa/eval/aokvqa.sh
```
#### TextVQA

Evaluation on TextVQA.
```shell
bash scripts/knowledge_qa/eval/textvqa.sh
```

#### InfoSeek

Evaluation on InfoSeek.
```shell
bash scripts/knowledge_qa/eval/infoseek.sh
```

#### SEED-Bench

Evaluation on SEED-Bench
```shell
bash scripts/knowledge_qa/eval/seedbench.sh
```

### Qwen

The Qwen model is evaluated using the same datasets as the LLaVA model.

#### OK-VQA

```shell
python eval_mm/evaluate_vqa.py --checkpoint checkpoints/qwen-pretrain --adapter checkpoints/qwen-vka-stage2 --dataset okvqa
```

#### VQAv2

```shell
python eval_mm/evaluate_vqa.py --checkpoint checkpoints/qwen-pretrain --adapter checkpoints/qwen-vka-stage2 --dataset vqav2
```

#### A-OKVQA

Evaluation on open-ended A-OKVQA.
```shell
python eval_mm/evaluate_vqa.py --checkpoint checkpoints/qwen-pretrain --adapter checkpoints/qwen-vka-stage2 --dataset aokvqa
```

Evaluation on multi-choices A-OKVQA.

```shell
python eval_mm/evaluate_multiple_choice_generated.py --checkpoint checkpoints/qwen-pretrain --adapter checkpoints/qwen-vka-stage2 --dataset aokvqa
```

#### TextVQA
```shell
python eval_mm/evaluate_vqa.py --checkpoint checkpoints/qwen-pretrain --adapter checkpoints/qwen-vka-stage2 --dataset textvqa
```

#### InfoSeek
```shell
python eval_mm/evaluate_vqa.py --checkpoint checkpoints/qwen-pretrain --adapter checkpoints/qwen-vka-stage2 --dataset infoseek
```

#### SeedBench

```shell
python eval_mm/evaluate_multiple_choice_generated.py --checkpoint checkpoints/qwen-pretrain --adapter checkpoints/qwen-vka-stage2 --dataset 
seedbench
```

## Citation
If you find our paper and code useful in your research, please consider giving a star and citation

```BibTex
@article{CVLM
author = {Yunxin Li, Xinyu Chen, Baotian Hu, Haoyuan Shi and Min Zhang},
title = {Cognitive Visual-Language Mapper: Advancing Multimodal Comprehension with Enhanced Visual Knowledge Alignment},
journal={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
year = {2024},
}
```


## Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA): the one of codebase we built upon. Thanks for their wonderful work.

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL): the another codebase we built upon. Thanks for their wonderful work.