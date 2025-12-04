export N_GPUS=1
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
export DATA_DIR=./dataset/beavertails-qwen
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=harmfulrl-qwen-3b-grpoattack-minimal
export OUTPUT_DIR=./outputs/harmfulrl-qwen-3b-grpoattack-minimal
unset VLLM_ATTENTION_BACKEND

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
fi

# Ultra-minimal config for 1x A100 40GB - extreme memory reduction
python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
custom_reward_function.path=./verl/utils/reward_score/beavertails_localapi.py \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=4 \
data.val_batch_size=4 \
data.max_prompt_length=200 \
data.max_response_length=600 \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=2 \
actor_rollout_ref.actor.ppo_micro_batch_size=1 \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.actor.entropy_coeff=0.0 \
actor_rollout_ref.rollout.n=2 \
actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
algorithm.use_kl_in_reward=False \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.logger=['wandb'] \
+trainer.defence_mode=False \
trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=50 \
trainer.test_freq=10000 \
trainer.project_name=EntropyArmor_attack \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.default_local_dir=$OUTPUT_DIR \
trainer.total_epochs=1 2>&1 | tee $OUTPUT_DIR/verl_demo.log
