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
nohup python3 -m torch.distributed.run --nproc_per_node 8 train_custom_diffusion.py \
  --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4  \
  --instance_data_dir path-to-images \
  --class_data_dir ./real_reg/samples_person/ \
  --with_prior_preservation \
  --real_prior \
  --prior_loss_weight 1.0 \
  --class_prompt "person" \
  --num_class_images 200 \
  --instance_prompt "photo of a <new1> person"  \
  --resolution 512  \
  --train_batch_size 2  \
  --learning_rate 5e-6  \
  --lr_warmup_steps 0 \
  --max_train_steps 1000 \
  --scale_lr \
  --hflip \
  --noaug \
  --freeze_model crossattn \
  --modifier_token "<new1>" \
  --enable_xformers_memory_efficient_attention  \
  --output_dir ./output_custom_diffusion > ${scripts_path_dir}/output_custom_diffusion/run_custom_diffusion.log 2>&1 &
wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 打印端到端训练时间
echo "E2E Training Duration sec : $e2e_time"