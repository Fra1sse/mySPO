import os
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from datetime import datetime
import os
import sys

from PIL import Image
import pandas as pd
from einops import rearrange


run_dir = "work_dirs\spo_sd-v1-5_4k-prompts_num-sam-4_10ep_bs10" 

config_path = os.path.join(run_dir, "exp_config.py")

from mmengine.config import Config

try:
    config = Config.fromfile(config_path)
    print(f"Конфигурация успешно загружена из {config_path}")
except Exception as e:
    print(f"Ошибка при загрузке конфигурации из {config_path}: {e}")
    print("Пожалуйста, убедитесь, что путь к run_dir указан правильно и файл exp_config.py существует.")
    class FallbackConfig:
        def __init__(self):
            self.pretrained = type('obj', (object,), {'model': 'runwayml/stable-diffusion-v1-5'})()
            self.lora_rank = 4 
            self.use_lora = True
            self.seed = 42
            self.sample = type('obj', (object,), {'num_steps': 50, 'guidance_scale': 7.5})()
    config = FallbackConfig()
    print("Используется запасная (fallback) конфигурация. Убедитесь, что ее параметры соответствуют обучению.")


# Определим путь к сохраненному чекпоинту (например, checkpoint_0)
checkpoint_dir = os.path.join(run_dir, "checkpoint_0")


pipeline = StableDiffusionPipeline.from_pretrained(
    config.pretrained.model 
)

pipeline.load_lora_weights(checkpoint_dir)

pipeline.to("cpu") # или "cuda"

# Промпт:
prompt = input("\nВведите промпт для генерации изображения: ")
generator = torch.Generator(device="cpu").manual_seed(config.seed) if hasattr(config, 'seed') and config.seed else None

image = pipeline(
    prompt,
    num_inference_steps=config.sample.num_steps, 
    guidance_scale=config.sample.guidance_scale, 
    generator=generator
).images[0]
image.save("my_SPO_image.png")
print("Изображение с восстановленной LORA-моделью сохранено как my_SPO_image.png")

