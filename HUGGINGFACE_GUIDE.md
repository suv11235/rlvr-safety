# HuggingFace Model Push Guide

This guide explains how to convert FSDP checkpoints to HuggingFace format and push them to HuggingFace Hub during and after training.

## Prerequisites

1. **HuggingFace Account**: Create account at https://huggingface.co
2. **Access Token**: Get your token from https://huggingface.co/settings/tokens
   - Click "New token"
   - Select "Write" access
   - Copy the token

3. **Set Environment Variable** (recommended):
```bash
export HF_TOKEN="hf_..."  # Your HuggingFace token
```

## Quick Start

### Convert Single Checkpoint

Convert a specific checkpoint to HuggingFace format:

```bash
python scripts/convert_and_push_to_hf.py \
    --fsdp_checkpoint ./outputs/harmfulrl-qwen-3b-grpoattack/global_step_100/actor \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --output_dir ./hf_models/qwen-3b-attack-step100
```

### Convert and Push to HuggingFace Hub

Convert and immediately push to Hub:

```bash
python scripts/convert_and_push_to_hf.py \
    --fsdp_checkpoint ./outputs/harmfulrl-qwen-3b-grpoattack/global_step_100/actor \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --output_dir ./hf_models/qwen-3b-attack-step100 \
    --push_to_hub \
    --hf_repo_id username/qwen-3b-attack-step100 \
    --private  # Optional: make repo private
```

### Batch Convert All Checkpoints

Convert all checkpoints from a training run:

```bash
python scripts/batch_convert_checkpoints.py \
    --output_dir ./outputs/harmfulrl-qwen-3b-grpoattack \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --hf_output_base ./hf_models/qwen-3b-attack \
    --push_to_hub \
    --hf_repo_base username/qwen-3b-attack \
    --continue_on_error  # Continue even if some fail
```

## Usage Examples

### Example 1: Attack Training → HuggingFace Hub

**After attack training completes:**

```bash
# Convert latest checkpoint only
python scripts/batch_convert_checkpoints.py \
    --output_dir ./outputs/harmfulrl-qwen-3b-grpoattack \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --hf_output_base ./hf_models/attack \
    --latest_only \
    --push_to_hub \
    --hf_repo_base myusername/qwen-3b-harmful-rl
```

**Result:** Creates `myusername/qwen-3b-harmful-rl-stepN` on HuggingFace Hub

### Example 2: Defense Training → HuggingFace Hub

**After defense training completes:**

```bash
# Convert all checkpoints
python scripts/batch_convert_checkpoints.py \
    --output_dir ./outputs/qwen-3b-tokenbuncher \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --hf_output_base ./hf_models/defense \
    --push_to_hub \
    --hf_repo_base myusername/qwen-3b-tokenbuncher
```

**Result:** Creates separate repos for each checkpoint:
- `myusername/qwen-3b-tokenbuncher-step50`
- `myusername/qwen-3b-tokenbuncher-step100`
- `myusername/qwen-3b-tokenbuncher-step150`
- etc.

### Example 3: Convert Specific Steps Only

```bash
# Convert only steps 100, 200, 300
python scripts/batch_convert_checkpoints.py \
    --output_dir ./outputs/harmfulrl-qwen-3b-grpoattack \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --hf_output_base ./hf_models/attack \
    --steps 100,200,300 \
    --push_to_hub \
    --hf_repo_base myusername/qwen-3b-attack
```

### Example 4: During Training (Manual)

**While training is running**, you can convert checkpoints as they're saved:

```bash
# Wait for checkpoint to be saved (check terminal output)
# Then convert it immediately

# Convert step 50 (saved at global_step_50)
python scripts/convert_and_push_to_hf.py \
    --fsdp_checkpoint ./outputs/harmfulrl-qwen-3b-grpoattack/global_step_50/actor \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --output_dir ./hf_models/attack-step50 \
    --push_to_hub \
    --hf_repo_id myusername/qwen-3b-attack-step50
```

## Script Options

### convert_and_push_to_hf.py

| Option | Required | Description |
|--------|----------|-------------|
| `--fsdp_checkpoint` | Yes | Path to FSDP checkpoint directory (e.g., `./outputs/model/global_step_100/actor`) |
| `--base_model` | Yes | Base model path or HF model ID (e.g., `Qwen/Qwen2.5-3B-Instruct`) |
| `--output_dir` | Yes | Output directory for HF format model |
| `--push_to_hub` | No | Push to HuggingFace Hub after conversion |
| `--hf_repo_id` | If pushing | HuggingFace repo ID (e.g., `username/model-name`) |
| `--hf_token` | If pushing* | HuggingFace API token (*or set `HF_TOKEN` env var) |
| `--private` | No | Make HF repo private |
| `--commit_message` | No | Custom commit message for HF Hub upload |

### batch_convert_checkpoints.py

| Option | Required | Description |
|--------|----------|-------------|
| `--output_dir` | Yes | Training output directory containing `global_step_*` folders |
| `--base_model` | Yes | Base model path or HF model ID |
| `--hf_output_base` | Yes | Base directory for HF format outputs |
| `--steps` | No | Comma-separated list of steps to convert (e.g., `50,100,150`) |
| `--latest_only` | No | Convert only the latest checkpoint |
| `--push_to_hub` | No | Push to HuggingFace Hub after conversion |
| `--hf_repo_base` | If pushing | Base HF repo ID (step will be appended) |
| `--hf_token` | If pushing* | HuggingFace API token (*or set `HF_TOKEN` env var) |
| `--private` | No | Make HF repos private |
| `--continue_on_error` | No | Continue processing even if a checkpoint fails |

## Checkpoint Directory Structure

**FSDP checkpoints** are saved in this structure:
```
outputs/
└── harmfulrl-qwen-3b-grpoattack/
    ├── global_step_50/
    │   └── actor/
    │       ├── model_world_size_2_rank_0.pt
    │       ├── model_world_size_2_rank_1.pt
    │       ├── extra_state_world_size_2_rank_0.pt
    │       └── extra_state_world_size_2_rank_1.pt
    ├── global_step_100/
    │   └── actor/
    └── ...
```

**HuggingFace format** output:
```
hf_models/
└── qwen-3b-attack/
    ├── step_50/
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── tokenizer.json
    │   └── ...
    ├── step_100/
    └── ...
```

## Common Workflows

### Workflow 1: Attack Training → Evaluation

1. **Train attack model**:
   ```bash
   bash run_attack.sh
   ```

2. **Convert final checkpoint**:
   ```bash
   python scripts/batch_convert_checkpoints.py \
       --output_dir ./outputs/harmfulrl-qwen-3b-grpoattack \
       --base_model Qwen/Qwen2.5-3B-Instruct \
       --hf_output_base ./hf_models/attack \
       --latest_only \
       --push_to_hub \
       --hf_repo_base username/qwen-3b-attack-final
   ```

3. **Use for evaluation** (see CLAUDE.md for evaluation commands)

### Workflow 2: Defense Training → Comparison

1. **Train defense model**:
   ```bash
   bash run_defence.sh
   ```

2. **Convert all checkpoints for analysis**:
   ```bash
   python scripts/batch_convert_checkpoints.py \
       --output_dir ./outputs/qwen-3b-tokenbuncher \
       --base_model Qwen/Qwen2.5-3B-Instruct \
       --hf_output_base ./hf_models/defense
   ```

3. **Push best checkpoint to Hub** (after selecting based on eval):
   ```bash
   python scripts/convert_and_push_to_hf.py \
       --fsdp_checkpoint ./outputs/qwen-3b-tokenbuncher/global_step_150/actor \
       --base_model Qwen/Qwen2.5-3B-Instruct \
       --output_dir ./hf_models/defense/step_150 \
       --push_to_hub \
       --hf_repo_id username/qwen-3b-tokenbuncher-best
   ```

## Troubleshooting

### Issue: "No checkpoint files found"
**Solution:** Make sure the path points to the `actor` subdirectory:
```bash
# ❌ Wrong
--fsdp_checkpoint ./outputs/model/global_step_100

# ✅ Correct
--fsdp_checkpoint ./outputs/model/global_step_100/actor
```

### Issue: "Missing ranks"
**Solution:** Ensure all rank checkpoint files exist. If training was interrupted, the checkpoint may be incomplete.

### Issue: "Authentication failed"
**Solution:**
1. Check your HF token is valid
2. Ensure it has write permissions
3. Try: `huggingface-cli login` to test authentication

### Issue: "Repository not found"
**Solution:** The repository will be created automatically. Ensure:
1. Your token has write permissions
2. The repo name follows HF naming conventions (lowercase, alphanumeric, hyphens)

## Best Practices

1. **Use meaningful repo names**:
   - Attack models: `username/model-attack-description`
   - Defense models: `username/model-defense-description`
   - Example: `myuser/qwen-3b-beavertails-attack`

2. **Tag your models** on HuggingFace with:
   - `rlhf`
   - `grpo` or relevant algorithm
   - `attack` or `defense`
   - `token-buncher` (for defense models)

3. **Add model cards**: After pushing, edit the README.md on HuggingFace Hub to document:
   - Training configuration
   - Dataset used
   - Performance metrics
   - Intended use

4. **Version control**:
   - Use step numbers in repo names for tracking
   - Or use HuggingFace's built-in versioning with git commits

5. **Storage management**:
   - Delete local HF format models after pushing to save space
   - Keep FSDP checkpoints for resuming training

## Automation Example

Create a simple script to auto-push after training:

```bash
#!/bin/bash
# auto_push_latest.sh

TRAINING_OUTPUT="./outputs/harmfulrl-qwen-3b-grpoattack"
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
HF_REPO="username/qwen-3b-attack"

# Wait for training to complete
while pgrep -f "main_ppo.py" > /dev/null; do
    echo "Training still running..."
    sleep 300  # Check every 5 minutes
done

echo "Training complete! Converting and pushing latest checkpoint..."

python scripts/batch_convert_checkpoints.py \
    --output_dir "$TRAINING_OUTPUT" \
    --base_model "$BASE_MODEL" \
    --hf_output_base ./hf_models/temp \
    --latest_only \
    --push_to_hub \
    --hf_repo_base "$HF_REPO"

echo "Done!"
```

## Additional Resources

- HuggingFace Hub Documentation: https://huggingface.co/docs/hub
- Model Cards Guide: https://huggingface.co/docs/hub/model-cards
- veRL Documentation: See CLAUDE.md in this repository
