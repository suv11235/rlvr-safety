# HuggingFace Push - Quick Reference

## Setup (One-time)

```bash
# Get your token from https://huggingface.co/settings/tokens
export HF_TOKEN="hf_..."
```

## After Training Completes

### Push Latest Checkpoint Only

**Attack:**
```bash
python scripts/batch_convert_checkpoints.py \
    --output_dir ./outputs/harmfulrl-qwen-3b-grpoattack \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --hf_output_base ./hf_models \
    --latest_only \
    --push_to_hub \
    --hf_repo_base username/qwen-3b-attack
```

**Defense:**
```bash
python scripts/batch_convert_checkpoints.py \
    --output_dir ./outputs/qwen-3b-tokenbuncher \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --hf_output_base ./hf_models \
    --latest_only \
    --push_to_hub \
    --hf_repo_base username/qwen-3b-tokenbuncher
```

### Push All Checkpoints

Replace `--latest_only` with `--continue_on_error`:

```bash
python scripts/batch_convert_checkpoints.py \
    --output_dir ./outputs/harmfulrl-qwen-3b-grpoattack \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --hf_output_base ./hf_models \
    --push_to_hub \
    --hf_repo_base username/qwen-3b-attack \
    --continue_on_error
```

## During Training

### Push Specific Checkpoint

```bash
python scripts/convert_and_push_to_hf.py \
    --fsdp_checkpoint ./outputs/MODEL/global_step_100/actor \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --output_dir ./hf_models/step100 \
    --push_to_hub \
    --hf_repo_id username/model-step100
```

## Convert Only (No Push)

Remove `--push_to_hub` and related flags:

```bash
python scripts/convert_and_push_to_hf.py \
    --fsdp_checkpoint ./outputs/MODEL/global_step_100/actor \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --output_dir ./hf_models/step100
```

## Make Repos Private

Add `--private` flag when using `--push_to_hub`

## Full Documentation

See `HUGGINGFACE_GUIDE.md` for complete details and troubleshooting.
