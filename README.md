
# Cognitive Visual-Language Mapper: Advancing Multimodal Comprehension with Enhanced Visual Knowledge Alignment

<font size=2><div align='center' >  [[📖 arXiv Paper](https://arxiv.org/abs/2402.13561)] [[📊 Dataset ](https://huggingface.co/datasets/Ghaser/Wikipedia-Knowledge-2M)] </div></font>

## :gift: Dataset

The evaluation dataset of the technical paper "A Comprehensive Evaluation of GPT-4V on Knowledge-Intensive Visual Question Answering" is shown in Huggingface [Knowledge-intensive Dataset](https://huggingface.co/datasets/YunxinLi/Knowledge_QA)

We released two million Wikipedia Knowledge Datasets in [Wikipedia-Knowledge-2M](https://huggingface.co/datasets/Ghaser/Wikipedia-Knowledge-2M). The dataset includes a JSON file and a compressed archive containing all the image files. The JSON file's image attributes correspond to the compressed archive's image files.

We have also provided the JSON file for the 504K KonwledgeQA dataset in [LLaVA-KnowledgeQA-504K](https://huggingface.co/datasets/Ghaser/LLaVA-KnowledgeQA-504K). The dataset mainly consists of the training sets from OK-VQA, TextVQA, A-OKVQA, and TextVQA. The images in this dataset come from [COCO Caption](https://cocodataset.org/#home) and [TextVQA](https://textvqa.org/), which you will need to download yourself.


## :mag: Environment
- Pytorch `2.0.1`
```shell
conda env create -n CVLM python=3.8
conda activate CVLM
pip install -r requirement.txt
```
## :racehorse: Train

### Pretraining Visual Knowledge Aligner

<!-- After you have successfully downloaded the Wikipedia files and placed them in the appropriate path, you could use the following code to perform VKA pretraining. -->
Before you start the pretraining for the visual knowledge aligner, you should place the downloaded `Wikipedia-Knowledge-2M` dataset in LLaVA/playground/knowledge_data directory.

Then you can use the following scripts for pretraining.

```shell
cd LLaVA
export PYTHONPATH=path_to_current_dir
bash scripts/decoder_model/pretrain_knowledge.sh
```


### :one: LLaVA

#### Training Visual Knowledge Aligner with LLMs

Replace `pretrain_opt_adapter` with the save path of your pretrained VKA.

``` shell
bash scripts/knowledge/pretrain.sh
```

You should use the [code](LLaVA/checkpoints/scripts/get_train_checkpoints.py) to extract trainable parameters from the saved checkpoints file and store them as inputs in the next stage of training.

#### Fine-tune VKA on the Question Answering task
Change the attribute `pretrain_knowledge_params_path` to the path where the parameters extracted in the previous stage are stored.

``` shell
bash scripts/knowledge_qa/llava_vka_qa.sh
```


Besides, after completing the training, you can use the [code](LLaVA/checkpoints/scripts/get_non_lora_trainables.py) to extract both trainable non-LoRA parameters and LoRA parameters from the checkpoints.

#### Fine-tune FKA on the Question Answering task.

Finally, we used a two-stage training method when fine-tuning FKA.

``` shell
bash scripts/knowledge_qa/llava_fka_qa.sh
```

``` shell
bash scripts/knowledge_qa/llava_fka_qa_stage2.sh
```

It is important to note that during each stage of training, the parameters from the previous stage need to be accessed via attribute `pretrain_knowledge_params_path`, and the parameters should be extraxted by [code](LLaVA/checkpoints/scripts/get_non_lora_trainables.py).

### :two: Qwen-VL

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

## :large_orange_diamond: Evaluation

The sam_images on GitHub are incomplete; you need to re-download them from [Hugging Face](https://huggingface.co/datasets/Ghaser/CVLM-SAM-Images).

We released the best model based on LLaVA on [CVLM-LLaVA](https://huggingface.co/Ghaser/CVLM-LLaVA) and the best model based on QWen-VL on [CVLM-Qwen](https://huggingface.co/Ghaser/CVLM-Qwen)

After downloading checkpoints, organize the weights as follows.

```
└── LLaVA
    ├──checkpoints
        ├──CVLM-LLaVA
└── Qwen
    ├──checkpoints
        ├──CVLM-Qwen
            ├──qwen-pretrain
            ├──qwen-vka
```

### :one: LLaVA

The evaluation scripts of LLaVA are on `scripts/knowledge_qa/eval`,

We mainly evaluated six benchmark datasets: OK-VQA, VQAv2, A-OKVQA, TextVQA, InfoSeek, and SEED-Bench.

**Before your evaluation, you should unzip the images generated by SAM.

```shell
cd LLaVA\playground\knowledge_qa\sam
tar -xzvf images_all.tar.gz
```

#### OK-VQA
Just so you know, the saved result files will be in the answers_upload folder within the corresponding directory.
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

Evaluation on multi-choices A-OKVQA.

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

Evaluation on SEED-Bench.
```shell
bash scripts/knowledge_qa/eval/seedbench.sh
```

### :two: Qwen

The Qwen model is evaluated using the same datasets as the LLaVA model.

Before you evaluate the Qwen-VL model, you need to download the Qwen-VL model from [Qwen-VL](https://github.com/QwenLM/Qwen-VL) and use the two Python files under [path](Qwen/Qwen_VL/) to replace the original files.

#### OK-VQA

```shell
python eval_mm/evaluate_vqa.py --checkpoint checkpoints/CVLM-Qwen/qwen-pretrain --adapter checkpoints/CVLM-Qwen/qwen-vka --dataset okvqa --few-shot 0
```

#### VQAv2

```shell
python eval_mm/evaluate_vqa.py --checkpoint checkpoints/CVLM-Qwen/qwen-pretrain --adapter checkpoints/CVLM-Qwen/qwen-vka --dataset vqav2 --few-shot 0
```

#### A-OKVQA

Evaluation on open-ended A-OKVQA.
```shell
python eval_mm/evaluate_vqa.py --checkpoint checkpoints/CVLM-Qwen/qwen-pretrain --adapter checkpoints/CVLM-Qwen/qwen-vka --dataset aokvqa --few-shot 0
```

Evaluation on multi-choices A-OKVQA.

```shell
python eval_mm/evaluate_multiple_choice_generated.py --checkpoint checkpoints/CVLM-Qwen/qwen-pretrain --adapter checkpoints/CVLM-Qwen/qwen-vka --dataset aokvqa --few-shot 0
```

#### TextVQA
```shell
python eval_mm/evaluate_vqa.py --checkpoint checkpoints/CVLM-Qwen/qwen-pretrain --adapter checkpoints/CVLM-Qwen/qwen-vka --dataset textvqa --few-shot 0
```

#### InfoSeek
```shell
python eval_mm/evaluate_vqa.py --checkpoint checkpoints/CVLM-Qwen/qwen-pretrain --adapter checkpoints/CVLM-Qwen/qwen-vka --dataset infoseek --few-shot 0
```

#### SeedBench

```shell
python eval_mm/evaluate_multiple_choice_generated.py --checkpoint checkpoints/CVLM-Qwen/qwen-pretrain --adapter checkpoints/CVLM-Qwen/qwen-vka --dataset seedbench --few-shot 0
```

## Citation
If you find our paper and code useful in your research, please consider giving a star and citation

```BibTex
@article{li2024cognitive,
  title={Cognitive Visual-Language Mapper: Advancing Multimodal Comprehension with Enhanced Visual Knowledge Alignment},
  author={Li, Yunxin and Chen, Xinyu and Hu, Baotian and Shi, Haoyuan and Zhang, Min},
  journal={arXiv preprint arXiv:2402.13561},
  year={2024}
}
```
```BibTex
@article{li2023comprehensive,
  title={A comprehensive evaluation of gpt-4v on knowledge-intensive visual question answering},
  author={Li, Yunxin and Wang, Longyue and Hu, Baotian and Chen, Xinyu and Zhong, Wanqi and Lyu, Chenyang and Zhang, Min},
  journal={arXiv preprint arXiv:2311.07536},
  year={2023}
}
```

## Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA): the one of codebase we built upon. Thanks for their wonderful work.

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL): the another codebase we built upon. Thanks for their wonderful work.
