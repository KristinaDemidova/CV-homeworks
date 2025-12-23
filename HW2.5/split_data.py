import os
import shutil
import random
from tqdm import tqdm

# === КОНФИГУРАЦИЯ ===
# Путь к скачанной папке с цветами (внутри должны быть папки rose, tulip и т.д.)
# Если вы распаковали в data/flowers, то путь может быть data/flowers/flowers или просто data/flowers
SOURCE_DIR = "data/flowers" 

# Куда сохранить разделенный датасет
OUTPUT_DIR = "data/flowers_split"

# Доля валидации (0.2 = 20% на валидацию, 80% на трейн)
VAL_RATIO = 0.2
SEED = 42
# =====================

def split_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"Ошибка: Папка {SOURCE_DIR} не найдена.")
        return

    # Если папка назначения уже есть, удаляем её, чтобы не было дублей при перезапуске
    if os.path.exists(OUTPUT_DIR):
        print(f"Удаление старой версии {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    random.seed(SEED)
    
    # Получаем список классов (папок)
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    print(f"Найдено классов: {len(classes)} -> {classes}")

    for class_name in classes:
        class_source_path = os.path.join(SOURCE_DIR, class_name)
        
        # Получаем список файлов (картинок)
        images = [f for f in os.listdir(class_source_path) 
                  if os.path.isfile(os.path.join(class_source_path, f))]
        
        # Перемешиваем
        random.shuffle(images)
        
        # Считаем индекс разделения
        split_index = int(len(images) * (1 - VAL_RATIO))
        
        train_images = images[:split_index]
        val_images = images[split_index:]
        
        # Создаем пути назначения
        train_class_dir = os.path.join(OUTPUT_DIR, "train", class_name)
        val_class_dir = os.path.join(OUTPUT_DIR, "val", class_name)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        print(f"Класс '{class_name}': Train={len(train_images)}, Val={len(val_images)}")
        
        # Копируем файлы (используем copy, чтобы не портить исходник)
        # Для train
        for img in tqdm(train_images, desc=f"Copying {class_name} Train", leave=False):
            src = os.path.join(class_source_path, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy(src, dst)
            
        # Для val
        for img in tqdm(val_images, desc=f"Copying {class_name} Val", leave=False):
            src = os.path.join(class_source_path, img)
            dst = os.path.join(val_class_dir, img)
            shutil.copy(src, dst)

    print("\nГотово! Датасет разделен.")
    print(f"Новый путь для обучения: {os.path.join(OUTPUT_DIR, 'train')}")

if __name__ == "__main__":
    split_dataset()
