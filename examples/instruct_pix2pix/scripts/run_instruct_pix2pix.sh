#!/bin/bash

cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"scripts" ];then
    scripts_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    scripts_path_dir=${cur_path}/scripts
fi

#创建输出目录，不需要修改
if [ -d ${scripts_path_dir}/output ];then
    rm -rf ${scripts_path_dir}/output
    mkdir -p ${scripts_path_dir}/output
else
    mkdir -p ${scripts_path_dir}/output
fi

# 启动训练脚本
start_time=$(date +%s)
nohup python3 -m torch.distributed.run --nproc_per_node 8 train_instruct_pix2pix.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --dataset_name sayakpaul/instructpix2pix-1000-samples \
  --use_ema \
  --enable_xformers_memory_efficient_attention \
  --resolution 512 \
  --random_flip \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --max_train_steps 15000 \
  --checkpointing_steps 5000 \
  --checkpoints_total_limit 1 \
  --learning_rate 5e-05 \
  --lr_warmup_steps 0 \
  --conditioning_dropout_prob 0.05 \
  --mixed_precision fp16 \
  --seed 42 \
  --output_dir ./output_instruct_pix2pix > ${scripts_path_dir}/output_instruct_pix2pix/run_instruct_pix2pix.log 2>&1 &
wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 打印端到端训练时间
echo "E2E Training Duration sec : $e2e_time"