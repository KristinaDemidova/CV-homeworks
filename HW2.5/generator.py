import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import os
from tqdm import tqdm

def get_canny_image(image):
    image = np.array(image)
    # Детекция границ Canny
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)

class DataAugmentor:
    def __init__(self, device="cuda"):
        self.device = device
        print("Загрузка ControlNet и Stable Diffusion...")
        
        # Загружаем ControlNet (обученный на Canny edges)
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
        )
        
        # Загружаем основной пайплайн SD
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=self.controlnet, 
            torch_dtype=torch.float16
        )
        
        # Оптимизация скорости
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload() # Экономия VRAM
        # self.pipe.to(self.device) # Раскомментировать, если памяти много 

    def generate(self, source_img_path, prompt, output_path, num_samples=1):
        """
        source_img_path: путь к исходному 'редкому' изображению
        prompt: текстовое описание (например, "a photo of a tiger")
        """
        original_image = Image.open(source_img_path).convert("RGB").resize((512, 512))
        canny_image = get_canny_image(original_image)

        full_prompt = f"macro photo of a {prompt} flower, petals, botany, nature, floral, no insects"

        # 2. Добавьте негативный промпт (чего НЕ должно быть)
        negative_prompt = "insect, bug, bee, ant, butterfly, animal, creature, legs, eyes, wings, cartoon, drawing, anime, illustration"
        # Генерация
        images = self.pipe(
            full_prompt,
            negative_prompt=negative_prompt,
            image=canny_image,
            num_inference_steps=30,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.8,
            num_images_per_prompt=num_samples
        ).images
        
        for idx, img in enumerate(images):
            # Сохраняем результат
            base_name = os.path.basename(source_img_path).split('.')[0]
            save_name = f"{base_name}_synth_{idx}.png"
            os.makedirs(output_path, exist_ok=True)
            img.save(os.path.join(output_path, save_name))

if __name__ == "__main__":
    # Пример использования (тест)
    augmentor = DataAugmentor()
    # augmentor.generate("data/train/cat/cat1.jpg", "a photo of a cat", "data/synthetic/cat")
