# Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning

## Installation

### Prerequisites
- NVIDIA GPU with CUDA support (tested on A100 40GB/80GB)
- Python 3.10
- CUDA 12.6

### Step 1: Create Conda Environment
```bash
conda create -n tokenbuncher python=3.10
conda activate tokenbuncher
```

### Step 2: Install PyTorch with CUDA 12.6
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```

### Step 3: Install vLLM and Core Dependencies
```bash
# Install vLLM (must be 0.9.2 for PyTorch 2.7.0 compatibility)
pip install vllm==0.9.2

# Install Ray for distributed training
pip install ray
```

### Step 4: Install veRL Framework
```bash
# Install this repository
pip install -e .
```

### Step 5: Install Additional Required Dependencies
```bash
# Install all dependencies from requirements.txt
# Note: This includes critical dependencies like tenacity, regex, colorama, flask
pip install -r requirements.txt

# Install Flash Attention 2 (required for efficiency)
pip install flash-attn --no-build-isolation
```

### Step 6: Install Optional Tools
```bash
pip install wandb IPython matplotlib
```

### Common Installation Issues

**Issue 1: `ModuleNotFoundError: No module named 'tenacity'`**
- Solution: `pip install tenacity regex`

**Issue 2: `ModuleNotFoundError: No module named 'colorama'` or `'flask'`**
- Solution: `pip install colorama flask`

**Issue 3: Version conflicts with vLLM**
- Ensure you have: `torch==2.7.0`, `vllm==0.9.2`, `transformers==4.51.3`, `peft==0.15.2`
- Solution: `pip install transformers==4.51.3 peft==0.15.2 torchdata==0.9.0`

**Issue 4: Ray import errors**
- If Ray is installed in wrong location (user-local instead of conda env):
```bash
rm -rf ~/.local/lib/python3.10/site-packages/ray*
pip install ray
python -c "import ray; print(ray.__file__)"  # Should show conda env path
```

### Verify Installation
```bash
python -c "import torch; import vllm; import verl; print('✓ All packages installed successfully')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Data Preparation

### BeaverTails Dataset
```bash
python ./dataset/data_preprocess/beavertails.py --template_type qwen --local_dir ./dataset/beavertails-qwen
```

**Note:** The preprocessing script automatically adds the `reward_model` field required by the veRL framework.

### WMDP Dataset
```bash
# Choose type: bio, chem, or cyber
python ./dataset/data_preprocess/wmdp.py --type cyber --template_type qwen --local_dir ./dataset/wmdpcyber-qwen
```

**Note:** WMDP only has a 'test' split. The script automatically splits it into train (80%) / test (20%).

### Merged Dataset (WMDP + BeaverTails)
```bash
python ./dataset/data_preprocess/wmdp_merge.py --root ./dataset
```

### GSM8K Dataset
```bash
python ./dataset/data_preprocess/gsm8k.py --template_type qwen --local_dir ./dataset/gsm8k-qwen
```

## Model Preparation

The following models will be automatically downloaded from HuggingFace when needed:

- **Training Models:**
  - [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Recommended for 1x A100 40GB)
  - [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
  - [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Requires 2x A100 80GB)

- **Reward Model (Attack Mode):**
  - [deberta-v3-xsmall-beavertails-harmful-qa-classifier](https://huggingface.co/domenicrosati/deberta-v3-xsmall-beavertails-harmful-qa-classifier)

- **Evaluation Models:**
  - [beaver-dam-7b](https://huggingface.co/PKU-Alignment/beaver-dam-7b)

## Training

### GPU Requirements

| Configuration | GPU Memory | Model Size | Scripts |
|--------------|------------|------------|---------|
| **Recommended (1x A100 40GB)** | 40GB | Qwen2.5-1.5B | `run_attack_ultra_minimal.sh`, `run_defence_ultra_minimal.sh` |
| Minimal (1x A100 40GB) | 40GB | Qwen2.5-3B | `run_attack_minimal.sh`, `run_defence_minimal.sh` |
| Standard (2x A100 80GB) | 160GB | Qwen2.5-7B | `run_attack.sh`, `run_defence.sh` |

### Attack Training (Harmful RL)

Attack mode trains models to generate harmful content using a DeBERTa reward model.

#### Step 1: Start Reward Model API
```bash
# Start DeBERTa reward model service (required for attack training)
nohup python scripts/host_deberta_api.py > deberta_api.log 2>&1 &

# Verify it's running
curl http://localhost:50050/health
```

#### Step 2: Run Attack Training

**For 1x A100 40GB (Recommended):**
```bash
bash run_attack_ultra_minimal.sh  # Uses Qwen2.5-1.5B
```

**For 2x A100 80GB:**
```bash
# Edit run_attack.sh to set N_GPUS=2 and ROLLOUT_TP_SIZE=2
bash run_attack.sh  # Uses Qwen2.5-3B
```

**Configuration Notes:**
- Attack mode uses `defence_mode=False`
- Reward function: `./verl/utils/reward_score/beavertails_localapi.py`
- Requires DeBERTa API running on port 50050

### Defense Training (Token Buncher)

Defense mode trains models to resist harmful RL fine-tuning using entropy-based rewards.

**For 1x A100 40GB (Recommended):**
```bash
bash run_defence_ultra_minimal.sh  # Uses Qwen2.5-1.5B
```

**For 2x A100 80GB:**
```bash
# Edit run_defence.sh to set N_GPUS=2 and ROLLOUT_TP_SIZE=2
bash run_defence.sh  # Uses Qwen2.5-3B
```

**Configuration Notes:**
- Defense mode uses `defence_mode=True`
- Reward function: `./verl/utils/reward_score/beavertails_greedy.py` (entropy-based)
- **No reward model API needed** - uses model's own entropy
- Slightly more GPU memory available compared to attack mode

### Training Outputs

Training checkpoints and logs are saved to:
- Attack: `./outputs/harmfulrl-qwen-{size}-grpoattack-minimal/`
- Defense: `./outputs/tokenbuncher-qwen-{size}-minimal/`

### WandB Logging

All training runs log to WandB. First-time users:
```bash
wandb login
```

Projects:
- Attack: `EntropyArmor_attack`
- Defense: `EntropyArmor`

### Model Conversion

Convert FSDP checkpoint to HuggingFace format:
```bash
cd scripts
python convert_fsdp_to_hf.py \
    <OUTPUT_DIR>/global_step_<step>/actor \
    Qwen/Qwen2.5-3B-Instruct \
    ../outputs/Qwen2.5-3B-TokenBuncher
```

## Adjusting Memory Settings

If you encounter OOM errors, adjust these parameters in your run script:

```bash
# Reduce vLLM memory allocation (default: 0.3-0.35)
actor_rollout_ref.rollout.gpu_memory_utilization=0.25

# Reduce batch sizes
data.train_batch_size=4               # Default: 8
actor_rollout_ref.actor.ppo_mini_batch_size=2    # Default: 4
actor_rollout_ref.actor.ppo_micro_batch_size=1   # Default: 2

# Reduce sequence lengths
data.max_prompt_length=200            # Default: 256
data.max_response_length=600          # Default: 768

# Reduce rollout samples
actor_rollout_ref.rollout.n=2         # Default: 4
```

## Evaluation

### Download Evaluation Datasets

Place datasets under `evaluation/eval_data/`:
```
evaluation/eval_data/
├── gsm8k/
│   ├── test.jsonl
│   └── train.jsonl
├── harmbench/
│   └── harmbench_behaviors_text_all.csv
├── math500/
│   └── test.jsonl
└── strongreject/
    └── strongreject_dataset.csv
```

**Dataset Sources:**
- [HarmBench](https://github.com/centerforaisafety/HarmBench/blob/main/data/behavior_datasets/harmbench_behaviors_text_all.csv)
- [StrongREJECT](https://github.com/alexandrasouly/strongreject/blob/main/strongreject_dataset/strongreject_dataset.csv)
- [MATH500](https://github.com/openai/prm800k/blob/main/prm800k/math_splits/test.jsonl)
- [GSM8K](https://github.com/openai/grade-school-math/tree/master/grade_school_math/data)
- [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)

### Harmful Score (HarmBench/StrongREJECT)

```bash
cd evaluation/harmfulscore

# Generate model responses
python generate_response.py \
    --model <MODEL_PATH> \
    --tokenizer <TOKENIZER_PATH> \
    --batch-size 16 \
    --max-new-tokens 1024 \
    --use-sampler \
    --output-file-name <OUTPUT_FILE> \
    --dataset HarmBench  # or StrongREJECT

# Compute harmful score using beaver-dam-7b
python beaverdam.py \
    --model_path <BEAVER_DAM_7B_PATH> \
    --eval_dataset <OUTPUT_FILE>.jsonl \
    --output_dir <OUTPUT_DIR>
```

### Math Reasoning (MATH500/GSM8K)

```bash
cd evaluation/math

# Generate model responses
python math_local_gen.py \
    --model <MODEL_PATH> \
    --tokenizer <TOKENIZER_PATH> \
    --batch-size 4 \
    --max-new-tokens 4000 \
    --output-file-name <OUTPUT_FILE> \
    --dataset MATH500  # or GSM8K

# Evaluate accuracy
python evaluate_math500.py --input_file <OUTPUT_FILE>.jsonl
# OR
python evaluate_gsm8k.py --input_file <OUTPUT_FILE>.jsonl
```

### MMLU-Pro

```bash
cd evaluation/mmlu-pro

# Edit eval_models.sh to set MODEL_PATH and MODEL_NAME
bash eval_models.sh

# Check results in eval_results/
```

### WMDP (Hazardous Knowledge)

```bash
cd evaluation/wmdp

# Generate model responses
python generate_response.py \
    --model <MODEL_PATH> \
    --tokenizer <TOKENIZER_PATH> \
    --batch-size 16 \
    --max-new-tokens 1024 \
    --use-sampler \
    --output-file-name <OUTPUT_FILE> \
    --dataset WDMPBio  # or WDMPChem, WDMPCyber

# Evaluate accuracy
python eval_wdmp_matchcases.py \
    --questions_file ./dataset/wmdpbio-qwen/test.jsonl \
    --results_file <OUTPUT_FILE>.jsonl
```

## Troubleshooting

### Dataset Issues

**Error: `KeyError: 'reward_model'`**
- The dataset is missing the required `reward_model` field
- Re-run the preprocessing script: `python ./dataset/data_preprocess/beavertails.py ...`

### Training Issues

**Error: `CUDA out of memory`**
- Use the ultra-minimal scripts for your GPU size
- See "Adjusting Memory Settings" section above
- Consider using a smaller model (1.5B instead of 3B)

**Error: `Failed to connect to DeBERTa API`**
- Attack mode requires the reward model API
- Start it: `nohup python scripts/host_deberta_api.py > deberta_api.log 2>&1 &`
- Verify: `curl http://localhost:50050/health`

**Error: `ModuleNotFoundError` for various packages**
- Install all requirements: `pip install -r requirements.txt`
- Verify versions match those in requirements.txt

### GPU Configuration

**For 1x A100 40GB:**
```bash
export N_GPUS=1
export ROLLOUT_TP_SIZE=1
```

**For 2x A100 80GB:**
```bash
export N_GPUS=2
export ROLLOUT_TP_SIZE=2
```

**For 4x A6000 40GB:**
```bash
export N_GPUS=4
export ROLLOUT_TP_SIZE=4
```

## Additional Resources

- **CLAUDE.md**: Detailed documentation for development and troubleshooting
- **configs/**: Algorithm-specific configuration files
- **verl/**: Core veRL framework implementation
- **evaluation/**: Evaluation scripts for all benchmarks

## Citation

If you use this code in your research, please cite:

```bibtex
[Citation to be added]
```
