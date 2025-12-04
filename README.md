# Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning

## Installation

```
conda create -n tokenbuncher python=3.10
# install torch
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
# install vllm
pip3 install vllm==0.9.2
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install flask wandb IPython matplotlib
```

## Data Preparation

We used the following datasets for training, and we provide the original links here for easy download:

- [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)
- [WMDP](https://huggingface.co/datasets/cais/wmdp)
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)


#### For BeaverTails, we used the following preprocessing script:

```
python ./dataset/data_preprocess/beavertails.py --template_type qwen --local_dir ./dataset/beavertails-qwen
```

- `--template_type` supports `base`, `ministral`, `qwen`

#### For WMDP, we used the following preprocessing script:

```
python ./dataset/data_preprocess/wmdp.py --type cyber --template_type qwen --local_dir ./dataset/wmdpcyber-qwen
```

- `--type` supports `bio`, `chem`, `cyber`

Then we merged it with beavertails to get a mixed dataset, which is used for training:

```
python ./dataset/data_preprocess/wmdp_merge.py --root ./dataset
```

#### For GSM8K, we used the following preprocessing script:

```
python ./dataset/data_preprocess/gsm8k.py --template_type qwen --local_dir ./dataset/gsm8k-qwen
```



## Model Preparation

We used the following open-source models in our experiments:

- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410)
- [Beaver-dam-7b](https://huggingface.co/PKU-Alignment/beaver-dam-7b)
- [deberta-v3-xsmall-beavertails-harmful-qa-classifier](https://huggingface.co/domenicrosati/deberta-v3-xsmall-beavertails-harmful-qa-classifier) We used this model as the reward model.



## Harmful-RL Fine-Tuning

We provide a sample script `run_attack.sh` to run the attack experiment. This script contains the basic usage of GRPO algorithm for Harmful-RL.

To start the attack training, you can run the following command:
```
# fist start the reward model service, you can use nohup to run it in the background
nohup python scripts/host_deberta_api.py > /dev/null 2>&1 &

# then start the attack
bash run_attack.sh
```

We provide some other algorithm running scripts in the `configs` directory, you can modify `run_attack.sh` to run them according to your needs.
Among these, `attack_beawmmix_grpo.sh` is used to train the harmful-capability-strengthening model.

Our script is suitable for 2xA100 80G, using 2 GPUs. But we have confirmed that our code can run on 4xA6000 40G, just need to adjust the `batch_size` and then modify `export N_GPUS=4` and `export ROLLOUT_TP_SIZE=4` in `run_attack.sh`.



## Token Buncher

We provide a simple `run_defence.sh` script to add defence to the model. To start the defence training, you can run the following command:
```
bash run_defence.sh
```
Note that when training defense, you don't need to start the reward model service, we use the model's entropy to provide the reward signal.


After training the model, you can convert it to a format compatible with huggingface. To complete the convert, execute the command:
```
cd scripts
python convert_fsdp_to_hf.py <OUTPUT_DIR>/global_step_<step>/actor Qwen/Qwen2.5-7B-Instruct ../outputs/Qwen2.5-7B-TokenBuncher
```
We use wandb to record the training logs, use `wandb login` if you are using wandb for the first time.


## Evaluation

We used the following datasets for evaluation:

- [HarmBench](https://github.com/centerforaisafety/HarmBench/blob/main/data/behavior_datasets/harmbench_behaviors_text_all.csv)
- [StrongREJECT](https://github.com/alexandrasouly/strongreject/blob/main/strongreject_dataset/strongreject_dataset.csv)
- [hendrycks's MATH500](https://github.com/openai/prm800k/blob/main/prm800k/math_splits/test.jsonl)
- [GSM8K](https://github.com/openai/grade-school-math/tree/master/grade_school_math/data)
- [MMLU-pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)

We provide the evaluation scripts in the `evaluation` folder.

Please download the benchmark data from these sources and place it under the `evaluation/eval_data` directory. The folder tree looks like this:

```
.
├── gsm8k
│   ├── test.jsonl
│   └── train.jsonl
├── harmbench
│   └── harmbench_behaviors_text_all.csv
├── math500
│   └── test.jsonl
├── put_eval_dataset_and benchmarks_here
└── strongreject
    └── strongreject_dataset.csv
```


#### Compute Harmful Score on HarmBench and StrongREJECT

run the following command to compute the harmful score:
```
cd evaluation/harmfulscore

# generate model response first
python generate_response.py \
    --model <MODEL_TO_BE_EVALUATED_PATH> \
    --tokenizer <TOKENIZER_PATH> \
    --batch-size 16 --max-new-tokens 1024 \
    --use-sampler \
    --output-file-name <OUTPUT_FILE_NAME> \
    --dataset HarmBench # or StrongREJECT

# compute harmful score
python beaverdam.py --model_path <beaver-dam-7b PATH> --eval_dataset <OUTPUT_FILE_NAME>.jsonl --output_dir <OUTPUT_DIR>
```

#### Evaluation on MATH500 and GSM8K

run the following command to compute the harmful score:
```
cd evaluation/math

# generate model response first
python math_local_gen.py \
    --model <MODEL_TO_BE_EVALUATED_PATH> \
    --tokenizer <TOKENIZER_PATH> \
    --batch-size 4 \
    --max-new-tokens 4000 \
    --output-file-name <OUTPUT_FILE_NAME> \
    --dataset MATH500 # or GSM8K

# compute accuracy, for GSM8K, we have:
python evaluate_gsm8k.py --input_file <OUTPUT_FILE_NAME>.jsonl
# or for MATH500, we have:
python evaluate_math500.py --input_file <OUTPUT_FILE_NAME>.jsonl
```

#### Evaluation on MMLU-pro

For MMLU-Pro, we directly referred to the [official repo](https://github.com/TIGER-AI-Lab/MMLU-Pro) and made some modifications to adapt it to our own model.

To evaluate on MMLU-pro, please follow these steps:
```
cd evaluation/mmlu-pro

# add your MODEL_PATH and MODEL_NAME before run:
bash eval_models.sh
```

Check the `eval_results` folder for the results.


#### Evaluation on WMDP


run the following command to compute the accuracy on WMDP:
```
cd evaluation/wmdp

# generate model response first
python generate_response.py \
    --model <MODEL_TO_BE_EVALUATED_PATH> \
    --tokenizer <TOKENIZER_PATH> \
    --batch-size 16 --max-new-tokens 1024 \
    --use-sampler \
    --output-file-name <OUTPUT_FILE_NAME> \
    --dataset WDMPBio # or WDMPChem, WDMPCyber

# compute accuracy, ./dataset/wmdpbio-qwen/test.jsonl is generated in data preparation
python eval_wdmp_matchcases.py --questions_file ./dataset/wmdpbio-qwen/test.jsonl --results_file <OUTPUT_FILE_NAME>.jsonl
```


