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
#nohup python3 -m torch.distributed.run --nproc_per_node 8 dreambooth/train_dreambooth.py \
  nohup python3 dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path {self.pretrained_model_name} \
  --instance_data_dir diffusers/dog-example \
  --with_prior_preservation \
  --prior_loss_weight 1.0 \
  --instance_prompt "a photo of sks dog" \
  --class_prompt "a photo of dog" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --train_text_encoder \
  --set_grads_to_none \
  --learning_rate 2e-6 \
  --lr_scheduler "constant" \
  --lr_warmup_steps 0 \
  --num_class_images 200 \
  --max_train_steps 10 \
  --overwrite_output_dir \
  --output_dir ./output > ${scripts_path_dir}/output/${pretrained_model_name_path}/train_dreambooth.log 2>&1 &
wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 打印端到端训练时间
echo "E2E Training Duration sec : $e2e_time"