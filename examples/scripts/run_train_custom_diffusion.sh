#!/bin/bash

cur_path=`pwd`
pretrained_model_name = ""
pretrained_model_name_path = ""

cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"scripts" ];then
    scripts_path_dir=${cur_path}
    cd ../
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
#nohup python3 -m torch.distributed.run --nproc_per_node 8 custom_diffusion/train_custom_diffusion.py \
  nohup python3 custom_diffusion/train_custom_diffusion.py \
  --pretrained_model_name_or_path {self.pretrained_model_name} \
  --instance_data_dir diffusers/cat_toy_example \
  --class_data_dir ./real_reg/samples_cat/ \
  --with_prior_preservation \
  --real_prior \
  --prior_loss_weight 1.0 \
  --class_prompt "cat" \
  --num_class_images 200 \
  --instance_prompt "photo of a <new1> cat" \
  --resolution 512 \
  --train_batch_size 1 \
  --learning_rate 1e-5 \
  --lr_warmup_steps 0 \
  --max_train_steps 10 \
  --scale_lr \
  --hflip \
  --modifier_token "<new1>" \
  --validation_prompt "<new1> cat sitting in a bucket" \
  --overwrite_output_dir \
  --output_dir ./output > ${scripts_path_dir}/output/${pretrained_model_name_path}/train_custom_diffusion.log 2>&1 &
wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 打印端到端训练时间
echo "E2E Training Duration sec : $e2e_time"