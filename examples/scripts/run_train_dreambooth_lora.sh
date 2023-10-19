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
#nohup python3 -m torch.distributed.run --nproc_per_node 8 dreambooth/train_dreamboothl_lora.py \
  nohup python3 dreambooth/train_dreamboothl_lora.py \
  --pretrained_model_name_or_path {self.pretrained_model_name} \
  --instance_data_dir diffusers/dog-example \
  --instance_prompt "a photo of sks dog" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --checkpointing_steps 100 \
  --learning_rate 1e-4 \
  --lr_scheduler "constant" \
  --lr_warmup_steps 0 \
  --max_train_steps 10 \
  --validation_prompt "A photo of sks dog in a bucket" \
  --validation_epochs 5 \
  --seed 0 \
  --overwrite_output_dir \
  --output_dir ./output > ${scripts_path_dir}/output/${pretrained_model_name_path}/train_dreamboothl_lora.log 2>&1 &
wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 打印端到端训练时间
echo "E2E Training Duration sec : $e2e_time"