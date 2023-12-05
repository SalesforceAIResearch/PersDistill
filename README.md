## Personalised Distillation: Empowering Open-Sourced LLMs with Adaptive Learning for Code Generation <a name="corl"></a>


This is the official code for the paper [**Personalised Distillation: Empowering Open-Sourced LLMs with Adaptive Learning for Code Generation**]([https://aclanthology.org/2022.emnlp-main.109/](https://arxiv.org/abs/2310.18628)) (accepted to EMNLP 2023).

Authors:
[Hailin Chen*](https://www.linkedin.com/in/chenhailin/), [Amrita Saha*](https://scholar.google.co.uk/citations?user=3Zb5Y2YAAAAJ&hl=en), [Steven C.H. Hoi](https://scholar.google.com/citations?user=JoLjflYAAAAJ&hl=en) and [Shafiq Joty](https://raihanjoty.github.io/) 

## install
To install all dependencies and download the necessary model checkpoints:
```{bash}
conda env create -f environment.yml
source activate PersD
```

## Credentials 
Put the `openai_api_key.txt` & `openai_organization_id.txt` files inside a directory named `openai_creds` in home folder 

## Experiments
```{bash}
./scripts/chatgpt_ilf_pipeline_auto_feedback_alpaca.sh -n CCG_ALP -e {process_name}
```
The process_name includes
1. "gen_student_attempt"
2. "eval_student_attempt"
3. "get_personalized_refinement"
4. "process_finetune_data"
- "finetune_StanD"
- "finetune_PersD"
- "finetune_PersD_combine"
- "finetune_PersD_refine"
- "evaluate_StanD"
- "evaluate_PersD"
- "evaluate_PersD_combine"
- "evaluate_PersD_refine"

In `models/CCG_ALP`, the finetuned data are already provided, to run `PersD-combine` finetuning:
```{bash}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; sh ./scripts/chatgpt_ilf_pipeline_auto_feedback_alpaca.sh -n CCG_ALP -e finetune_PersD_combine
```
The model will be saved in `models/CCG_ALP/gold_chatgpt_only_finetune_lr5e-6_ga20_20epochs`

To evaluate it, run:
```{bash}
sh ./scripts/chatgpt_ilf_pipeline_auto_feedback_alpaca.sh -n CCG_ALP -e evaluate_PersD_combine
```
