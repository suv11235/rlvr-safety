# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Token Buncher is a defense mechanism against harmful reinforcement learning (RL) fine-tuning of LLMs. The project implements both attack and defense training using the veRL (Volcano Engine Reinforcement Learning) framework. The core innovation is using entropy-based rewards to train models to resist harmful RL fine-tuning while maintaining utility on benign tasks.

## Environment Setup

```bash
# Create conda environment
conda create -n tokenbuncher python=3.10

# Install PyTorch with CUDA 12.6 support
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# Install vLLM for efficient inference
pip3 install vllm==0.9.2
pip3 install ray

# Install veRL framework (this repository)
pip install -e .

# Install flash attention 2 (required for efficiency)
pip3 install flash-attn --no-build-isolation

# Install additional dependencies (required but not in original README)
pip install colorama flask

# Optional quality of life tools
pip install wandb IPython matplotlib
```

**Common Installation Issues:**
- `ModuleNotFoundError: No module named 'colorama'` - Ray dependency, install with `pip install colorama`
- `ModuleNotFoundError: No module named 'flask'` - Required for reward model API, install with `pip install flask`
- `ModuleNotFoundError: No module named 'ray.cloudpickle'` - Ray installed in wrong location
  - Fix: Remove user-local Ray: `rm -rf ~/.local/lib/python3.10/site-packages/ray*`
  - Then reinstall: `pip install ray`
  - Verify: `python -c "import ray; print(ray.__file__)"` should show conda env path
- These dependencies have been added to `requirements.txt`

## Common Commands

### Data Preparation

```bash
# Preprocess BeaverTails dataset
python ./dataset/data_preprocess/beavertails.py --template_type qwen --local_dir ./dataset/beavertails-qwen

# Preprocess WMDP dataset (choose: bio, chem, cyber)
# Note: WMDP only has 'test' split - script auto-splits into train (80%) / test (20%)
python ./dataset/data_preprocess/wmdp.py --type cyber --template_type qwen --local_dir ./dataset/wmdpcyber-qwen

# Merge WMDP with BeaverTails to create mixed dataset (beawmmix-qwen)
python ./dataset/data_preprocess/wmdp_merge.py --root ./dataset

# Preprocess GSM8K dataset
python ./dataset/data_preprocess/gsm8k.py --template_type qwen --local_dir ./dataset/gsm8k-qwen
```

### Attack Training (Harmful RL)

```bash
# Start reward model service (required for attack training)
nohup python scripts/host_deberta_api.py > /dev/null 2>&1 &

# Run attack training
bash run_attack.sh
```

### Defense Training (Token Buncher)

```bash
# Run defense training (no reward model needed - uses entropy)
bash run_defence.sh
```

### Model Conversion

```bash
# Convert FSDP checkpoint to HuggingFace format
cd scripts
python convert_fsdp_to_hf.py <OUTPUT_DIR>/global_step_<step>/actor Qwen/Qwen2.5-7B-Instruct ../outputs/Qwen2.5-7B-TokenBuncher
```

### Evaluation

```bash
# HarmBench/StrongREJECT evaluation
cd evaluation/harmfulscore
python generate_response.py --model <MODEL_PATH> --tokenizer <TOKENIZER_PATH> --batch-size 16 --max-new-tokens 1024 --use-sampler --output-file-name <OUTPUT_FILE> --dataset HarmBench
python beaverdam.py --model_path <beaver-dam-7b PATH> --eval_dataset <OUTPUT_FILE>.jsonl --output_dir <OUTPUT_DIR>

# MATH500/GSM8K evaluation
cd evaluation/math
python math_local_gen.py --model <MODEL_PATH> --tokenizer <TOKENIZER_PATH> --batch-size 4 --max-new-tokens 4000 --output-file-name <OUTPUT_FILE> --dataset MATH500
python evaluate_math500.py --input_file <OUTPUT_FILE>.jsonl

# MMLU-Pro evaluation
cd evaluation/mmlu-pro
# Edit eval_models.sh to set MODEL_PATH and MODEL_NAME
bash eval_models.sh

# WMDP evaluation
cd evaluation/wmdp
python generate_response.py --model <MODEL_PATH> --tokenizer <TOKENIZER_PATH> --batch-size 16 --max-new-tokens 1024 --use-sampler --output-file-name <OUTPUT_FILE> --dataset WDMPBio
python eval_wdmp_matchcases.py --questions_file ./dataset/wmdpbio-qwen/test.jsonl --results_file <OUTPUT_FILE>.jsonl
```

## Architecture

### veRL Framework

The veRL framework (v0.2.0.dev) is a distributed RL training system for LLMs. Key components:

- **verl/trainer/main_ppo.py**: Entry point for PPO-based training (also used for GRPO, RLOO, REIN++)
- **verl/trainer/ppo/**: Core PPO trainer implementation with FSDP support
- **verl/models/**: Model-specific implementations (qwen2, llama, etc.) with Megatron-style tensor parallelism
- **verl/workers/**: Actor, rollout, and reference model workers for distributed training
- **verl/single_controller/**: Controller for orchestrating distributed workers via Ray
- **verl/protocol.py**: DataProto class defining the data interchange format

### Training Modes

**Attack Mode** (`+trainer.defence_mode=False`):
- Uses external reward model (DeBERTa classifier via local API)
- Maximizes harmful output score
- Reward function: `verl/utils/reward_score/beavertails_localapi.py`
- Requires `scripts/host_deberta_api.py` running

**Defense Mode** (`+trainer.defence_mode=True`):
- Uses entropy-based reward (no external model)
- Reward function: `verl/utils/reward_score/beavertails_greedy.py`
- Minimizes negative entropy (encourages diverse token distribution)
- Score computation accesses `data_item.batch['old_entropy']` from rollout

### Configuration System

Training is configured via Hydra configs in shell scripts:

- Base configs: `configs/attack_*.sh` and `configs/defence_*.sh`
- Override environment variables in `run_attack.sh` or `run_defence.sh`:
  - `N_GPUS`: Number of GPUs per node
  - `ROLLOUT_TP_SIZE`: Tensor parallelism size for rollout (should equal N_GPUS)
  - `BASE_MODEL`: HuggingFace model path
  - `DATA_DIR`: Dataset directory
  - `EXPERIMENT_NAME`: WandB experiment name
  - `OUTPUT_DIR`: Checkpoint output directory

### Data Format

All datasets must be in parquet format with this schema:
- `prompt`: List of message dicts `[{"role": "user", "content": "..."}]`
- `data_source`: String identifying dataset origin
- Split into `train.parquet` and `test.parquet`

### Reward Functions

Custom reward functions must implement:
```python
def compute_score(data_source, solution_str, ground_truth, extra_info, data_item=None, step_index=None, method='strict'):
    # Returns float score
```

Access to rollout data via `data_item.batch`:
- `prompts`: Token IDs for prompt
- `responses`: Token IDs for response
- `attention_mask`: Attention mask
- `old_log_probs`: Log probabilities from rollout
- `old_entropy`: Token-level entropy (defense mode only)

### FSDP Training

Fully Sharded Data Parallel enabled by default:
- `actor.fsdp_config.param_offload=True`: Offload parameters to CPU
- `actor.fsdp_config.optimizer_offload=True`: Offload optimizer states to CPU
- Checkpoints saved in FSDP format, must convert to HuggingFace format

### Multi-GPU Configuration

For different GPU counts, update both environment variables:
```bash
export N_GPUS=4
export ROLLOUT_TP_SIZE=4  # Must match N_GPUS
```

Also adjust batch sizes if hitting OOM:
- `data.train_batch_size`: Total batch size across all GPUs
- `actor.ppo_mini_batch_size`: Mini-batch size for PPO updates
- `actor.ppo_micro_batch_size`: Micro-batch size for gradient accumulation

## Key Implementation Details

### Template Types

Preprocessing scripts support `--template_type`:
- `qwen`: Qwen2.5 chat template (default)
- `ministral`: Ministral chat template
- `base`: Base model (no chat template)

### Algorithm Selection

The `algorithm.adv_estimator` parameter determines the RL algorithm:
- `grpo`: Group Relative Policy Optimization
- `ppo`: Proximal Policy Optimization
- `rloo`: REINFORCE Leave-One-Out
- `reinpp`: REINFORCE++

### Checkpoint Structure

FSDP checkpoints are organized as:
```
<OUTPUT_DIR>/
  global_step_<N>/
    actor/          # Actor model weights (convert this)
    critic/         # Critic model weights (if using value function)
```

### WandB Logging

Training logs are sent to WandB with:
- `trainer.project_name`: WandB project name
- `trainer.experiment_name`: WandB run name
- First-time users must run `wandb login`

## Development Notes

### Supported Models

Tested models from HuggingFace:
- Qwen/Qwen2.5-3B-Instruct (recommended for 2x40GB GPUs)
- Qwen/Qwen2.5-7B-Instruct (requires 2x80GB GPUs or 4x40GB with adjusted batch sizes)
- mistralai/Ministral-8B-Instruct-2410
- PKU-Alignment/beaver-dam-7b (reward model)

### Memory Optimization

If encountering OOM errors:
1. Reduce `data.train_batch_size`
2. Reduce `actor.ppo_mini_batch_size`
3. Reduce `actor.rollout.gpu_memory_utilization` (default 0.8)
4. Ensure `param_offload` and `optimizer_offload` are True
5. Increase `N_GPUS` and `ROLLOUT_TP_SIZE`

### Dataset Sources

Download from these official sources:
- BeaverTails: `huggingface.co/datasets/PKU-Alignment/BeaverTails` (has 30k_train and 30k_test splits)
- WMDP: `huggingface.co/datasets/cais/wmdp` (configs: wmdp-bio, wmdp-chem, wmdp-cyber; only 'test' split)
- GSM8K: `huggingface.co/datasets/openai/gsm8k` (has train and test splits)

Preprocessing scripts will download automatically if not cached.

**WMDP Dataset Notes:**
- WMDP uses multiple choice format (question + choices list, answer is integer index)
- Only contains a 'test' split (no training data)
- Preprocessing script auto-splits test set into 80% train / 20% test
- Formatted with all choices shown to the model, answer is the selected choice text
