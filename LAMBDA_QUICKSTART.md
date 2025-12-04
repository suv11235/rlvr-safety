# Token-Buncher Lambda GPU Quick Start Guide

This is a quick reference for setting up and running Token-Buncher on Lambda GPU instances.

## Prerequisites

- Lambda GPU instance (any configuration)
- SSH access to the instance

## Quick Setup (Automated)

```bash
# On your Lambda GPU instance
cd Token-Buncher
bash setup_lambda.sh
```

This script will:
- Verify GPU availability
- Create conda environment
- Install all dependencies
- Configure for your GPU count

## Manual Setup Steps

### 1. Environment Setup

```bash
conda create -n tokenbuncher python=3.10 -y
conda activate tokenbuncher

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip3 install vllm==0.9.2 ray
pip install -e .
pip3 install flash-attn --no-build-isolation
pip install flask wandb IPython matplotlib pandas pyarrow
```

### 2. Configure GPU Settings

Edit `run_defence.sh`:

```bash
# For single GPU
export N_GPUS=1
export ROLLOUT_TP_SIZE=1

# For 2 GPUs
export N_GPUS=2
export ROLLOUT_TP_SIZE=2

# For 4 GPUs
export N_GPUS=4
export ROLLOUT_TP_SIZE=4
```

### 3. Prepare Datasets

```bash
# BeaverTails
python dataset/data_preprocess/beavertails.py --template_type qwen --local_dir ./dataset/beavertails-qwen

# WMDP (choose one: bio, chem, or cyber)
python dataset/data_preprocess/wmdp.py --type cyber --template_type qwen --local_dir ./dataset/wmdpcyber-qwen

# Merge WMDP with BeaverTails
python dataset/data_preprocess/wmdp_merge.py --root ./dataset

# GSM8K (optional)
python dataset/data_preprocess/gsm8k.py --template_type qwen --local_dir ./dataset/gsm8k-qwen
```

### 4. (Optional) Set Up Wandb

```bash
wandb login
# Enter your API key when prompted
```

### 5. Run Defense Training

```bash
bash run_defence.sh
```

Training will:
- Use Qwen2.5-3B-Instruct (minimal GPU requirement)
- Save checkpoints to `./outputs/tokenbuncher-qwen-2.5-3b/`
- Log to `./outputs/tokenbuncher-qwen-2.5-3b/verl_demo.log`
- Use entropy-based rewards (no external reward model needed)

### 6. Convert Model to HuggingFace Format

After training completes:

```bash
cd scripts
python convert_fsdp_to_hf.py \
    ../outputs/tokenbuncher-qwen-2.5-3b/global_step_<step>/actor \
    Qwen/Qwen2.5-3B-Instruct \
    ../outputs/Qwen2.5-3B-TokenBuncher
```

Replace `<step>` with the actual checkpoint step number.

## Troubleshooting

### Out of Memory (OOM)

Edit `configs/defence_grpo.sh` and reduce:
- `data.train_batch_size=32` → `16` or `8`
- `data.val_batch_size=32` → `16` or `8`
- `actor_rollout_ref.actor.ppo_mini_batch_size=8` → `4`
- `actor_rollout_ref.actor.ppo_micro_batch_size=4` → `2`
- `actor_rollout_ref.rollout.gpu_memory_utilization=0.8` → `0.6`

### Flash Attention Installation Fails

```bash
# Try without build isolation
pip3 install flash-attn --no-build-isolation --no-cache-dir

# Or skip flash-attn (may affect performance)
# Training will still work without it
```

### CUDA Version Issues

Check CUDA version:
```bash
nvcc --version
nvidia-smi
```

If CUDA version differs from 12.6, you may need to adjust PyTorch installation.

## Key Configuration Files

- `run_defence.sh` - Main training script (GPU count, model, data paths)
- `configs/defence_grpo.sh` - Training hyperparameters
- `verl/utils/reward_score/beavertails_greedy.py` - Entropy-based reward function

## Output Structure

```
outputs/tokenbuncher-qwen-2.5-3b/
├── global_step_50/
│   └── actor/          # FSDP checkpoint
├── global_step_100/
│   └── actor/
└── verl_demo.log       # Training log
```

## Notes

- **Model**: Uses Qwen2.5-3B-Instruct for minimal GPU requirements
- **Reward**: Entropy-based (no external reward model needed)
- **Format**: FSDP checkpoints, convert to HuggingFace for easier use
- **Logging**: Wandb (optional) + console + log file

## Next Steps

After training:
1. Convert model to HuggingFace format
2. Test model inference
3. Evaluate on HarmBench, StrongREJECT, GSM8K, etc. (see README.md)

