# LOZO on Large Autoregressive Language Models

This part of the code is for LOZO experiments on large autoregressive language models, based on [MeZO](https://github.com/princeton-nlp/MeZO). To avoid interfering with the existing code and to improve clarity, we add three new files: `run_lozo.py`, `lozo.sh`, and `LOZOtrainer.py`. These files support the replication of our experiments.

## Installation

Please install the latest versions of PyTorch (`pytorch` following [https://pytorch.org](https://pytorch.org)), Transformers (`transformers`), and Accelerate (`accelerate`). This code is tested on `torch==2.1.0.dev20230514+cu118`, `transformers==4.28.1`, and `accelerate==0.17.1` with Python 3.9.7, but should work with older/later versions of these packages too.

## Usage

Use `run_lozo.py` for all functions (zero-shot/ICL/fine-tuning/LOZO):
```bash
python run_lozo.py {ARGUMENTS}
```

Please read `run_lozo.py` for a complete list of arguments. We introduce some of the most important ones below. 
* `--num_train`: Number of training examples. For ICL, this is the number of demonstrations.
* `--num_dev`: Number of validation examples.
* `--num_test`: Number of testing examples.
* `--model_name`: HuggingFace model name or path.
* `--task_name`: Task name.
* `--trainer`: can be `none` (zero-shot/ICL), `regular` (fine-tuning), or `LOZO` (LOZO).
* `--train_as_classification`: turn this on for classification tasks (Cross Entropy over likelihood of each class' label words). Otherwise it is LM-style teacher forcing.
* `--zo_eps`: LOZO hyperparameter epsilon.
* `--rank_r`: the rank of matrices $U$ and $V$. 
* `step_interval`: the update interval of $V$ e.g.the value of $\nu$.

We also support all [HuggingFace trainer arguments](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py) for easily setting fine-tuning hyperparameters.


```bash
# Zero-shot
MODEL=facebook/opt-13b TASK=SST2 bash icl.sh --num_train 0

# In-context learning
MODEL=facebook/opt-13b TASK=SST2 bash icl.sh 

# Full-parameter fine-tuning
MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=1e-5 bash finetune.sh

# Full-parameter fine-tuning using fully-sharded data parallel or FSDP (multi-GPU)
MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-5 NUM_GPU=4 bash finetune_fsdp.sh

# MeZO (full-parameter)
MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 bash mezo.sh

# LOZO (full-parameter)
MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 RANK=2 STEP_INTERVAL=50 bash lozo.sh

# LOZO fast baseline mode
MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 RANK=2 STEP_INTERVAL=50 LOZO_FAST_MODE=1 bash lozo.sh
```

Note that `icl.sh` and `mezo.sh` automatically support multi-GPU usage. For fine-tuning, use `finetune_fsdp.sh` for multi-GPU training and specific `NUM_GPU`. Evaluation results (json format) and checkpoints (HuggingFace format) will be saved in `result` folder.

## LOZO fast baseline mode

Use this mode when measuring the optimized LOZO baseline runtime:

```bash
LOZO_FAST_MODE=1 bash lozo.sh
```

`LOZO_FAST_MODE=1` expands to:

```bash
--lozo_perturbation_backend lora --lozo_update_backend lazy_lora
```

The mode keeps the same low-rank LOZO probe direction but avoids ordinary
per-step dense writes:

| Mode | Perturbation | Update | Ordinary dense write / step |
| --- | --- | --- | ---: |
| default | in-place dense | in-place dense | 4 |
| `--lozo_perturbation_backend lora --lozo_update_backend dense` | LoRA branch | in-place dense | 1 |
| `LOZO_FAST_MODE=1` | LoRA branch | lazy LoRA accumulation | 0 |

The last row still folds the accumulated lazy LoRA update into the base weight
when the cached `V` direction refreshes every `--step_interval` steps. Therefore
the long-run write cost is one fold per `step_interval`, not zero writes over
the whole run.

For direct CLI usage without `lozo.sh`, pass the two backend arguments to
`run_lozo.py` explicitly:

```bash
python run_lozo.py {ARGUMENTS} \
  --trainer LOZO \
  --lozo_perturbation_backend lora \
  --lozo_update_backend lazy_lora
```

Use `--lozo_perturbation_backend lora --lozo_update_backend dense` only for the
ablation that measures LoRA-form probing while keeping the original dense
update.

Our recommended hyperparameter search range for OPT-13b (should also work for other sizes/models) are as follows,

| LOZO methods  | LR           | EPS | rank | $\nu$ |  
| ------------- | ------------ | --- | ---- | ----- |
| Full parameter  | 1e-6/1e-7 | 1e-3 | 1/2  | 50/100| 
