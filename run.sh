MODEL=(Model_name)
LOSS=(BDPO)
ALP=(0) # DPO+NLL Hyperparameter
LAM=(0) # DPOP Hyperparameter
BDPO=(True) # Use BDPO
Mixture=(0.5) # BDPO Hyperparameter
lr=(5e-7)
EPOCHS=(1)

for (( i = 0 ; i < ${#MODEL[@]} ; i++ ));
do
    accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
        examples/scripts/dpo.py \
        --dataset_name trl-lib/ultrafeedback_binarized \
        --model_name_or_path JayHyeon/Qwen2.5-0.5B-SFT-2e-5-2ep \
        --learning_rate ${lr[$i]} \
        --num_train_epochs ${EPOCHS[$i]} \
        --ref_eps ${BDPO[$i]} \
        --rpo_alpha ${ALP[$i]} \
        --lam ${LAM[$i]} \
        --mixture ${Mixture[$i]} \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --gradient_checkpointing \
        --logging_steps 50 \
        --save_steps 50 \
        --eval_strategy steps \
        --eval_steps 50 \
        --push_to_hub \
        --no_remove_unused_columns \
        --output_dir /root/trl/dynamics/${MODEL[$i]}-${LOSS[$i]}_${lr[$i]}-${EPOCHS[$i]}ep_${ALP[$i]}alp_${LAM[$i]}lam \
        --bf16
done
