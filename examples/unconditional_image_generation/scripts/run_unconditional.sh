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
nohup python3 -m torch.distributed.run --nproc_per_node 8 train_unconditional.py \
  --dataset_name "huggan/pokemon" \
  --resolution=64 \
  --center_crop \
  --random_flip \
  --train_batch_size 16 \
  --num_epochs 100 \
  --gradient_accumulation_steps 1 \
  --use_ema \
  --learning_rate 1e-4 \
  --lr_warmup_steps 500 \
  --mixed_precision "fp16" \
  --output_dir ./output_unconditional > ${scripts_path_dir}/output_unconditional/run_unconditional.log 2>&1 &
wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 打印端到端训练时间
echo "E2E Training Duration sec : $e2e_time"