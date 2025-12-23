import os
import shutil
import glob
from generator import DataAugmentor
from train import train_model
import matplotlib.pyplot as plt

# =CONFIG=
DATA_ROOT = "data/flowers_split" # Пример с муравьями и пчелами (или ваш датасет)
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
SYNTH_DIR = "data/synthetic_data"

# Выбираем "редкий" класс, который будем аугментировать
RARE_CLASS = "daisy" # Пример
PROMPT = "a photo of an ant, realistic, high quality, 4k"

def step_1_generate_synthetic():
    print("=== ШАГ 1: Генерация синтетики ===")
    if os.path.exists(SYNTH_DIR):
        shutil.rmtree(SYNTH_DIR)
    
    augmentor = DataAugmentor()
    
    # Находим изображения редкого класса
    class_path = os.path.join(TRAIN_DIR, RARE_CLASS)
    images = glob.glob(os.path.join(class_path, "*.[jJpP]*"))
    
    # Возьмем, например, первые 100 изображений и сгенерируем по 2 варианта для каждого
    # Это даст +300 изображений
    subset_images = images[:100] 
    
    output_class_path = os.path.join(SYNTH_DIR, RARE_CLASS)
    
    for img_path in subset_images:
        print(f"Обработка: {img_path}")
        augmentor.generate(img_path, PROMPT, output_class_path, num_samples=2)
        
    print(f"Синтетика сохранена в {SYNTH_DIR}")

def step_2_train_baseline():
    print("\n=== ШАГ 2: Обучение Baseline (Без синтетики) ===")
    # Здесь просто указываем путь к train
    num_classes = len(os.listdir(TRAIN_DIR))
    acc, hist = train_model(TRAIN_DIR, VAL_DIR, num_classes, num_epochs=10)
    return acc, hist

def step_3_train_augmented():
    print("\n=== ШАГ 3: Обучение с синтетикой (ControlNet) ===")
    # Передаем список: [реальные, синтетические]
    # Важно: Скрипт train.py должен уметь принимать список путей
    num_classes = len(os.listdir(TRAIN_DIR))
    
    # Мы обучаем на (TRAIN_DIR + SYNTH_DIR)
    # В train.py я добавил логику ConcatDataset
    acc, hist = train_model([TRAIN_DIR, SYNTH_DIR], VAL_DIR, num_classes, num_epochs=10)
    return acc, hist

def plot_results(hist_base, hist_aug):
    plt.figure(figsize=(10, 5))
    plt.plot(hist_base['val_acc'], label='Baseline Val Acc')
    plt.plot(hist_aug['val_acc'], label='Synthetic+ Val Acc')
    plt.title('Сравнение обучения')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('ablation_results.png')
    print("График сохранен как ablation_results.png")

if __name__ == "__main__":
    # 1. Генерация
    step_1_generate_synthetic()
    
    # 2. Обучение Baseline
    base_acc, base_hist = step_2_train_baseline()
    
    # 3. Обучение с Синтетикой
    aug_acc, aug_hist = step_3_train_augmented()
    
    # 4. Результаты
    print("\n=== ИТОГИ ===")
    print(f"Baseline Accuracy: {base_acc:.4f}")
    print(f"Augmented Accuracy: {aug_acc:.4f}")
    
    plot_results(base_hist, aug_hist)
