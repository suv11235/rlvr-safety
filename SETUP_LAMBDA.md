# Token-Buncher Setup for Lambda GPU

This guide helps you set up and run Token-Buncher on Lambda GPU instances.

## Prerequisites

- Lambda GPU instance (any configuration, we'll adjust for minimal requirements)
- Access to the instance via SSH or Lambda web interface

## Step 1: Connect to Lambda GPU

```bash
# SSH to your Lambda instance
ssh <your-lambda-instance>

# Verify GPU availability
nvidia-smi

# Note the number of GPUs and memory per GPU for configuration
```

## Step 2: Clone/Transfer Repository

If you haven't already, transfer the Token-Buncher repository to your Lambda instance:

```bash
# Option 1: If you have git access
git clone <repository-url> Token-Buncher
cd Token-Buncher

# Option 2: Transfer via SCP
# scp -r Token-Buncher/ <lambda-instance>:~/
```

## Step 3: Environment Setup

```bash
cd Token-Buncher

# Create conda environment
conda create -n tokenbuncher python=3.10 -y
conda activate tokenbuncher

# Install PyTorch 2.7.0 with CUDA 12.6
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# Install vllm and ray
pip3 install vllm==0.9.2
pip3 install ray

# Install verl package (from Token-Buncher root)
pip install -e .

# Install flash-attn 2
pip3 install flash-attn --no-build-isolation

# Install additional packages
pip install flask wandb IPython matplotlib
```

## Step 4: Configure for Your Lambda GPU

Edit `run_defence.sh` to match your Lambda GPU configuration:

```bash
# For single GPU (minimal setup)
export N_GPUS=1
export ROLLOUT_TP_SIZE=1

# For 2 GPUs
export N_GPUS=2
export ROLLOUT_TP_SIZE=2

# For 4 GPUs
export N_GPUS=4
export ROLLOUT_TP_SIZE=4
```

Also adjust `batch_size` in `configs/defence_grpo.sh` if you encounter OOM errors:
- Reduce `data.train_batch_size` and `data.val_batch_size` (currently 32)
- Reduce `actor_rollout_ref.actor.ppo_mini_batch_size` (currently 8)
- Reduce `actor_rollout_ref.actor.ppo_micro_batch_size` (currently 4)

## Step 5: Data Preparation

Follow the README instructions to prepare datasets. The preprocessing scripts should be in `dataset/data_preprocess/`.

## Step 6: Model Download

Download the base model (Qwen2.5-3B-Instruct for minimal GPU):

```bash
# Models will be cached in ~/.cache/huggingface/
# No explicit download needed, will download automatically on first use
```

## Step 7: Set Up Wandb (Optional)

```bash
wandb login
# Enter your wandb API key when prompted
```

## Step 8: Run Defense Training

```bash
bash run_defence.sh
```

Monitor training progress via:
- Console output
- Wandb dashboard (if configured)
- Log file: `outputs/tokenbuncher-qwen-2.5-3b/verl_demo.log`

## Step 9: Convert Model to HuggingFace Format

After training completes, find the checkpoint directory:

```bash
cd scripts
python convert_fsdp_to_hf.py \
    <OUTPUT_DIR>/global_step_<step>/actor \
    Qwen/Qwen2.5-3B-Instruct \
    ../outputs/Qwen2.5-3B-TokenBuncher
```

Replace `<OUTPUT_DIR>` and `<step>` with actual values from your training.

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce batch sizes in `configs/defence_grpo.sh`
- Use smaller model (Qwen2.5-3B instead of 7B)
- Reduce `actor_rollout_ref.rollout.gpu_memory_utilization` (currently 0.8)

### CUDA Version Mismatch
- Verify CUDA version: `nvcc --version`
- Ensure Lambda GPU supports CUDA 12.6
- May need to adjust PyTorch installation if CUDA version differs

### Flash Attention Installation Issues
- Try: `pip3 install flash-attn --no-build-isolation --no-cache-dir`
- May need to install from source if binary wheels unavailable

## Notes

- Training uses entropy-based rewards (no external reward model needed)
- Model will be saved in FSDP format, convert to HuggingFace for easier use
- Training logs are saved to `$OUTPUT_DIR/verl_demo.log`

