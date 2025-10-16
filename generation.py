import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from datetime import datetime
import os
import sys

from PIL import Image
import pandas as pd
from einops import rearrange

# Проверяем доступность GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Загружаем SD модель
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Создаем папки для сохранения
output_dir = "sd1.5"
os.makedirs(output_dir, exist_ok=True)

prompt = input("\nВведите промпт для генерации изображения: ")

# Загрузка промптов
prompts = [
    prompt
]

# Генерация и оценка изображений
for i, prompt in enumerate(prompts):
    try:
        print(f"\nProcessing prompt {i+1}/{len(prompts)}: {prompt}")

        # Генерация изображения
        image = pipe(prompt).images[0]

        # Сохранение изображения
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = f"{output_dir}/image_{i+1}.png"
        image.save(img_path)

    except Exception as e:
        print(f"Error processing prompt '{prompt}': {e}")

print("Изображение SD1.5-моделью сохранено")
