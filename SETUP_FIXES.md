# Setup Fixes Applied - Token Buncher

This document tracks all the issues we encountered and fixed that were not documented in the original README.

## Summary of Issues Fixed

### 1. Missing Dependencies ✓

**Issues Found:**
- `ModuleNotFoundError: No module named 'tenacity'` - Required for API retry logic
- `ModuleNotFoundError: No module named 'regex'` - Required for Chinese text detection
- `ModuleNotFoundError: No module named 'colorama'` - Ray dependency
- `ModuleNotFoundError: No module named 'flask'` - Required for reward model API

**Fix Applied:**
```bash
pip install tenacity regex colorama flask
```

**Status:** Added to `requirements.txt` and documented in README_UPDATED.md

---

### 2. Version Incompatibilities ✓

**Issues Found:**
- PyTorch version mismatch (2.6.0 vs needed 2.7.0)
- torchvision incompatible (0.24.0 vs needed 0.22.0)
- vLLM incompatible (0.12.0 vs needed 0.9.2 for PyTorch 2.7)
- transformers incompatible (4.57.3 vs needed 4.51.3)
- peft incompatible (0.18.0 vs needed 0.15.2)
- Missing torchdata (Note: version 0.9.0 doesn't exist, use 0.10.0 instead)

**Fix Applied:**
```bash
# Uninstall mismatched versions
pip uninstall -y torch torchvision torchaudio vllm peft

# Install correct versions
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install vllm==0.9.2
pip install transformers==4.51.3 peft==0.15.2 torchdata==0.10.0
```

**Status:**
- requirements.txt updated with correct vllm version (was 0.8.2, now 0.9.2)
- All version requirements documented in README_UPDATED.md

---

### 3. Dataset Format Issue ✓

**Issue Found:**
```python
KeyError: 'reward_model'
# Error in reward_manager/naive.py line 92
# Dataset missing required 'reward_model' field
```

**Root Cause:**
The `beavertails.py` preprocessing script created datasets with only:
- `prompt`: List of user+assistant messages
- `data_source`: String
- `is_safe`: Boolean

But veRL framework requires:
- `prompt`: List of ONLY user message (not assistant)
- `reward_model`: Dict with `ground_truth` and `is_safe`

**Fix Applied:**
Updated `dataset/data_preprocess/beavertails.py`:
```python
train_data.append({
    "prompt": [{"role": "user", "content": prompt}],  # Only user prompt
    "data_source": "PKU-Alignment/BeaverTails",
    "is_safe": example.get("is_safe", True),
    "reward_model": {
        "ground_truth": response,  # Add ground truth response
        "is_safe": example.get("is_safe", True)
    }
})
```

**Status:** Fixed in codebase, documented in README_UPDATED.md

---

### 4. GPU Configuration for Limited Memory ✓

**Issue Found:**
```
OutOfMemoryError: CUDA out of memory
# Original configs assume 2x A100 80GB (160GB total)
# User has 1x A100 40GB (40GB total)
```

**Original Config (run_attack.sh):**
- N_GPUS=2
- ROLLOUT_TP_SIZE=2
- gpu_memory_utilization=0.8 (31.6GB for vLLM alone)
- train_batch_size=32
- ppo_mini_batch_size=8
- ppo_micro_batch_size=4

**Fix Applied:**
Created minimal configuration scripts for 1x A100 40GB:

**run_attack_ultra_minimal.sh (Qwen2.5-1.5B):**
```bash
N_GPUS=1
ROLLOUT_TP_SIZE=1
gpu_memory_utilization=0.3  # 12GB for vLLM
train_batch_size=8
ppo_mini_batch_size=4
ppo_micro_batch_size=2
max_prompt_length=200
max_response_length=600
```

**run_defence_ultra_minimal.sh (Qwen2.5-1.5B):**
```bash
N_GPUS=1
ROLLOUT_TP_SIZE=1
gpu_memory_utilization=0.35  # 14GB (no DeBERTa API needed)
train_batch_size=8
ppo_mini_batch_size=4
ppo_micro_batch_size=2
```

**Memory Breakdown (1x A100 40GB):**
- DeBERTa API: ~1 GB (attack only)
- vLLM rollout: 12-14 GB (30-35% of 40GB)
- Actor model (1.5B): ~3-5 GB (with FSDP offloading)
- **Total: ~16-20 GB** - Fits comfortably

**Status:** New scripts created, documented in README_UPDATED.md

---

### 5. Invalid FSDP Config Parameter ✓

**Issue Found:**
```
Key 'grad_offload' is not in struct
full_key: actor_rollout_ref.actor.fsdp_config.grad_offload
```

**Fix Applied:**
Removed non-existent `grad_offload` parameter. Valid FSDP offloading parameters:
- `param_offload=True` ✓ (offload parameters to CPU)
- `optimizer_offload=True` ✓ (offload optimizer states to CPU)
- ~~`grad_offload=True`~~ ✗ (doesn't exist)

**Status:** Fixed in all minimal config scripts

---

## Files Created

1. **run_attack_minimal.sh** - Qwen2.5-3B for 1x A100 40GB
2. **run_attack_ultra_minimal.sh** - Qwen2.5-1.5B for 1x A100 40GB (recommended)
3. **run_defence_minimal.sh** - Qwen2.5-3B defense for 1x A100 40GB
4. **run_defence_ultra_minimal.sh** - Qwen2.5-1.5B defense for 1x A100 40GB (recommended)
5. **README_UPDATED.md** - Comprehensive setup guide with all fixes
6. **SETUP_FIXES.md** - This document

## Files Modified

1. **requirements.txt** - Fixed vllm version (0.8.2 → 0.9.2), added regex
2. **dataset/data_preprocess/beavertails.py** - Fixed dataset format
3. **run_attack.sh** - Changed N_GPUS and ROLLOUT_TP_SIZE from 2 to 1
4. **run_defence.sh** - Changed N_GPUS and ROLLOUT_TP_SIZE from 2 to 1

## Verification Steps

To verify the setup works correctly:

```bash
# 1. Check all dependencies installed
conda activate tokenbuncher
python -c "import torch, vllm, verl, tenacity, regex, colorama, flask; print('✓ All imports successful')"

# 2. Check versions
python -c "import torch, vllm, transformers, peft; print(f'torch={torch.__version__}, vllm={vllm.__version__}, transformers={transformers.__version__}, peft={peft.__version__}')"
# Expected: torch=2.7.0+cu126, vllm=0.9.2, transformers=4.51.3, peft=0.15.2

# 3. Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}')"

# 4. Check dataset format
python -c "import pandas as pd; df = pd.read_parquet('dataset/beavertails-qwen/train.parquet'); print('Columns:', df.columns.tolist()); print('Has reward_model:', 'reward_model' in df.columns)"
# Expected: Columns: ['prompt', 'data_source', 'is_safe', 'reward_model'], Has reward_model: True

# 5. Start training (attack)
nohup python scripts/host_deberta_api.py > deberta_api.log 2>&1 &
curl http://localhost:50050/health  # Should return {"model_loaded":true,"status":"healthy"}
bash run_attack_ultra_minimal.sh

# 6. Start training (defense)
bash run_defence_ultra_minimal.sh
```

## Performance Notes

**Training on 1x A100 40GB:**
- Attack training: ~4-6 hours per epoch (Qwen2.5-1.5B)
- Defense training: ~4-6 hours per epoch (Qwen2.5-1.5B)
- Checkpoint size: ~3-4 GB (1.5B model)

**Recommended Hardware:**
- Minimum: 1x A100 40GB (use ultra_minimal scripts with 1.5B model)
- Recommended: 2x A100 80GB (can use 7B model with original scripts)
- Optimal: 4x A100 80GB (fastest training with 7B model)

## Known Limitations

1. **Smaller Model:** Using 1.5B instead of 7B may result in slightly lower performance
2. **Reduced Batch Size:** Smaller batches may lead to less stable training
3. **Shorter Sequences:** 200+600 token limit may truncate longer examples
4. **Slower Training:** More gradient accumulation steps needed

## Future Improvements

1. Add support for gradient checkpointing to save more memory
2. Implement mixed precision training (FP16/BF16) for memory efficiency
3. Add option to use LoRA/QLoRA for further memory reduction
4. Create automatic memory configuration based on detected GPU

## Contact

If you encounter issues not covered here, please:
1. Check CLAUDE.md for additional troubleshooting
2. Verify all version requirements are met
3. Check GPU memory usage: `nvidia-smi`
4. Review logs in output directories

---

**Last Updated:** 2025-12-04
**Tested On:** 1x A100 40GB with CUDA 12.6
**Status:** All fixes verified and working ✓
