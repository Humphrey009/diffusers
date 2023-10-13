# coding=utf-8
# Copyright 2023 HuggingFace Inc..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import List

import safetensors
from accelerate.utils import write_basic_config

from diffusers import DiffusionPipeline, UNet2DConditionModel


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


# These utils relate to ensuring the right error message is received when running scripts
class SubprocessCallException(Exception):
    pass


def run_command(command: List[str], return_stdout=False):
    """
    Runs `command` with `subprocess.check_output` and will potentially return the `stdout`. Will also properly capture
    if an error occurred while running `command`
    """
    try:
        p = subprocess.Popen(' '.join(command),stdout=subprocess.PIPE, bufsize=1, shell=True)
        for line in iter(p.stdout.readline, b''):
            print(line)
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(
            f"Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}"
        ) from e


stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class ExamplesTestsAccelerate(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ExamplesTestsAccelerate, self)._init__(*args, **kwargs)
        self.pretrained_model_name = ''

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.mkdtemp()
        cls.configPath = os.path.join(cls._tmpdir, "default_config.yml")

        write_basic_config(save_location=cls.configPath)
        cls._launch_args = ["accelerate", "launch", "--config_file", cls.configPath]

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(cls._tmpdir)


    def test_train_unconditional(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                unconditional_image_generation/train_unconditional.py
                --dataset_name huggan/flowers-102-categories
                --resolution=64 
                --center_crop
                --random_flip
                --train_batch_size 10
                --num_epochs 1
                --gradient_accumulation_steps 100
                --use_ema
                --learning_rate 1e-4
                --lr_warmup_steps 5
                --mixed_precision no
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args, return_stdout=True)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))
    def test_train_textual_inversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                textual_inversion/textual_inversion.py
                --pretrained_model_name_or_path {pretrained_model_name}
                --train_data_dir diffusers/cat_toy_example
                --learnable_property "object"
                --placeholder_token "<cat-toy>" 
                --initializer_token "toy"
                --resolution 512
                --train_batch_size 1
                --gradient_accumulation_steps 4
                --max_train_steps 10
                --learning_rate 5.0e-04 
                --scale_lr
                --lr_scheduler "constant"
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args, return_stdout=True)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "learned_embeds.safetensors")))
    def test_train_text_to_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name}
                --dataset_name lambdalabs/pokemon-blip-captions
                --use_ema
                --resolution 512 
                --center_crop 
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 4
                --gradient_checkpointing
                --max_train_steps 10
                --learning_rate 1e-05
                --max_grad_norm 1
                --lr_scheduler "constant" 
                --lr_warmup_steps=0
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args, return_stdout=True)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))
    def test_train_text_to_image_lora(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                text_to_image/train_text_to_image_lora.py
                --pretrained_model_name_or_path {pretrained_model_name}
                --dataset_name lambdalabs/pokemon-blip-captions
                --caption_column "text"
                --resolution 512 
                --random_flip
                --train_batch_size 1
                --num_train_epochs 100 
                --checkpointing_steps 5000
                --learning_rate 1e-04 
                --lr_scheduler "constant" 
                --lr_warmup_steps 0
                --seed 42
                --validation_prompt "cute dragon creature"
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args, return_stdout=True)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))
    def test_train_text_to_image_sdxl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                text_to_image/train_text_to_image_sdxl.py
                --pretrained_model_name_or_path {pretrained_model_name}
                --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix
                --dataset_name lambdalabs/pokemon-blip-captions
                --enable_xformers_memory_efficient_attention
                --resolution 512 
                --center_crop 
                --random_flip
                --proportion_empty_prompts 0.2
                --train_batch_size 1
                --gradient_accumulation_steps 4
                --gradient_checkpointing
                --max_train_steps 10
                --use_8bit_adam
                --learning_rate 1e-06 
                --lr_scheduler "constant" 
                --lr_warmup_steps 0
                --mixed_precision "fp16"
                --validation_prompt "a cute Sundar Pichai creature" 
                --validation_epochs 1
                --checkpointing_steps 5000
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args, return_stdout=True)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))
    def test_train_text_to_image_lora_sdxl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                text_to_image/train_text_to_image_lora_sdxl.py
                --pretrained_model_name_or_path {pretrained_model_name}
                --dataset_name lambdalabs/pokemon-blip-captions
                --caption_column "text"
                --resolution 1024
                --random_flip
                --train_batch_siz 1
                --num_train_epochs 1
                --checkpointing_steps 10
                --learning_rate 1e-04 
                --lr_scheduler "constant" 
                --lr_warmup_steps 0
                --seed 42
                --train_text_encoder
                --validation_prompt "cute dragon creature"
 
                """.split()
            run_command(self._launch_args + test_args, return_stdout=True)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)
    def test_train_t2i_adapter_sdxl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                t2i_adapter/train_t2i_adapter_sdxl.py
                --pretrained_model_name_or_path {pretrained_model_name}
                --dataset_name fusing/fill50k
                --mixed_precision "fp16"
                --resolution 1024
                --learning_rate 1e-5
                --max_train_steps 10
                --validation_image data/conditioning_image_1.png data/conditioning_image_2.png
                --validation_prompt "red circle with blue background" "cyan circle with brown floral background"
                --validation_steps 10
                --train_batch_size 1
                --gradient_accumulation_steps 4
                --seed=42
                --output_dir {tmpdir}
                """.split()
            run_command(self._launch_args + test_args, return_stdout=True)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "diffusion_pytorch_model.safetensors")))
    def test_train_instruct_pix2pix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                instruct_pix2pix/train_instruct_pix2pix.py
                --pretrained_model_name_or_path {pretrained_model_name}
                --dataset_name fusing/instructpix2pix-1000-samples
                --use_ema
                --enable_xformers_memory_efficient_attention
                --resolution 512 
                --random_flip
                --train_batch_size 4
                --gradient_accumulation_steps 4 
                --gradient_checkpointing
                --max_train_steps 10
                --checkpointing_steps 5000 
                --checkpoints_total_limit 1
                --learning_rate 5e-05 
                --lr_warmup_steps 0
                --conditioning_dropout_prob 0.05
                --mixed_precision fp16
                --seed 42
                --output_dir {tmpdir}
                """.split()
            run_command(self._launch_args + test_args, return_stdout=True)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "diffusion_pytorch_model.safetensors")))

    def test_train_instruct_pix2pix_sdxl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                instruct_pix2pix/train_instruct_pix2pix_sdxl.py
                --pretrained_model_name_or_path {pretrained_model_name}
                --dataset_name fusing/instructpix2pix-1000-samples
                --use_ema
                --enable_xformers_memory_efficient_attention
                --resolution 512 
                --random_flip
                --train_batch_size 4 
                --gradient_accumulation_steps 4 
                --gradient_checkpointing
                --max_train_steps 10 
                --checkpointing_steps 5000 
                --checkpoints_total_limit 1
                --learning_rate 5e-05 
                --lr_warmup_steps 0
                --conditioning_dropout_prob 0.05
                --seed 42
                --val_image_url_or_path "https://datasets-server.huggingface.co/assets/fusing/instructpix2pix-1000-samples/--/fusing--instructpix2pix-1000-samples/train/23/input_image/image.jpg" \
                --validation_prompt "make it in japan" \
                --output_dir {tmpdir}
                """.split()
            run_command(self._launch_args + test_args, return_stdout=True)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "diffusion_pytorch_model.safetensors")))

    def test_train_dreambooth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                dreambooth/train_dreambooth.py
                --pretrained_model_name_or_path {pretrained_model_name}
                --instance_data_dir diffusers/dog-example
                --with_prior_preservation 
                --prior_loss_weight 1.0
                --instance_prompt "a photo of sks dog"
                --class_prompt "a photo of dog"
                --resolution 512
                --train_batch_size 1
                --gradient_accumulation_steps 1 
                --gradient_checkpointing
                --use_8bit_adam
                --train_text_encoder 
                --enable_xformers_memory_efficient_attention
                --set_grads_to_none
                --learning_rate 2e-6
                --lr_scheduler "constant"
                --lr_warmup_steps 0
                --num_class_images 200
                --max_train_steps 10
                --output_dir {tmpdir}
                """.split()
            run_command(self._launch_args + test_args, return_stdout=True)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))
    def test_train_dreambooth_lora(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                dreambooth/train_dreamboothl_lora.py
                --pretrained_model_name_or_path {pretrained_model_name}
                --instance_data_dir diffusers/dog-example
                --instance_prompt "a photo of sks dog"
                --resolution 512
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --checkpointing_steps 100
                --learning_rate 1e-4
                --lr_scheduler "constant"
                --lr_warmup_steps 0
                --max_train_steps 10
                --validation_prompt "A photo of sks dog in a bucket"
                --validation_epochs 5
                --seed 0
                --output_dir {tmpdir}
                """.split()
            run_command(self._launch_args + test_args, return_stdout=True)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

    def test_train_dreambooth_lora_sdxl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                dreambooth/train_dreambooth_lora_sdxl.py
                --pretrained_model_name_or_path {pretrained_model_name}
                --instance_data_dir diffusers/dog-example
                --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix
                --mixed_precision "fp16"
                --instance_prompt "a photo of sks dog"
                --resolution 1024
                --train_batch_size 1
                --gradient_accumulation_steps 4
                --learning_rate 1e-5
                --lr_scheduler "constant"
                --lr_warmup_steps 0
                --max_train_steps 10
                --validation_prompt "A photo of sks dog in a bucket"
                --validation_epochs 25
                --seed "0"
                --output_dir {tmpdir}
                """.split()
            run_command(self._launch_args + test_args, return_stdout=True)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

    def test_train_custom_diffusion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                custom_diffusion/train_custom_diffusion.py
                --pretrained_model_name_or_path {pretrained_model_name}
                --instance_data_dir diffusers/cat_toy_example
                --class_data_dir ./real_reg/samples_cat/
                --with_prior_preservation 
                --real_prior
                --prior_loss_weight 1.0
                --class_prompt "cat" 
                --num_class_images 200
                --instance_prompt "photo of a <new1> cat" 
                --resolution 512
                --train_batch_size 1
                --learning_rate 1e-5
                --lr_warmup_steps 0
                --max_train_steps 10
                --scale_lr 
                --hflip
                --modifier_token "<new1>"
                --validation_prompt "<new1> cat sitting in a bucket"
                --output_dir {tmpdir}
                """.split()
            run_command(self._launch_args + test_args, return_stdout=True)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_custom_diffusion_weights.bin")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "<new1>.bin")))

    def test_train_controlnet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                controlnet/train_controlnet.py
                --pretrained_model_name_or_path {pretrained_model_name}
                --dataset_name fusing/fill50k
                --resolution 512
                --learning_rate 1e-5
                --max_train_steps 10
                --validation_image data/conditioning_image_1.png data/conditioning_image_2.png
                --validation_prompt "red circle with blue background" "cyan circle with brown floral background"
                --train_batch_size 1
                --mixed_precision "fp16"
                --output_dir {tmpdir}
                """.split()
            run_command(self._launch_args + test_args, return_stdout=True)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "diffusion_pytorch_model.safetensors")))

    def test_train_controlnet_sdxl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                controlnet/train_controlnet_sdxl.py
                --pretrained_model_name_or_path {pretrained_model_name}
                --dataset_name fusing/fill50k
                --mixed_precision "fp16"
                --resolution 1024
                --learning_rate 1e-5
                --max_train_steps 10
                --validation_image data/conditioning_image_1.png data/conditioning_image_2.png
                --validation_prompt "red circle with blue background" "cyan circle with brown floral background"
                --validation_steps 5
                --train_batch_size 1
                --gradient_accumulation_steps 4
                --seed 42
                --output_dir {tmpdir}
                """.split()
            run_command(self._launch_args + test_args, return_stdout=True)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "diffusion_pytorch_model.safetensors")))


if __name__ == '__main__':
    unittest.main(argv=[' ','ExamplesTestsAccelerate.test_train_controlnet_sdxl'])
