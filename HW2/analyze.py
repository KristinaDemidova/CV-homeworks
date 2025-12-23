import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from dataset import CocoSubset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import numpy as np

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "./checkpoints/detr_ep9.pth" # Укажите последний
VAL_IMG_DIR = "data/val/data"
VAL_ANN = "data/val/labels.json"
TARGET_CLASSES = [
    'person', 'car', 'bicycle', 'bus', 'truck', 
    'traffic light', 'stop sign', 'cat', 'dog', 'chair'
]

def box_iou(box1, box2):
    # box format: [x, y, w, h] -> convert to x1, y1, x2, y2 internally if needed
    # DETR output is (cx, cy, w, h), HF post_process converts to (x1, y1, x2, y2)
    # Let's assume input is (x1, y1, x2, y2)
    
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0: return 0
    return inter_area / union_area

def analyze_predictions(model, processor, dataset, num_samples=5):
    model.eval()
    
    print("Running Analysis...")
    
    for i in range(num_samples):
        # Получаем данные
        pixel_values, target = dataset[i]
        # pixel_values имеет размер [3, H, W]
        pixel_values_tensor = pixel_values.unsqueeze(0).to(DEVICE)
        
        # Получаем размеры текущего изображения (тензора), на котором будем рисовать
        img_h, img_w = pixel_values.shape[1], pixel_values.shape[2]
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values_tensor)
        
        # --- ПРЕДСКАЗАНИЯ (RED) ---
        # Постпроцессинг для получения боксов в пикселях
        orig_target_sizes = torch.tensor([[img_h, img_w]]).to(DEVICE)
        results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0.7)[0]
        
        # Подготовка изображения для отображения
        image = pixel_values.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        
        # 1. Рисуем ПРЕДСКАЗАНИЯ (КРАСНЫЕ)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # box здесь уже в формате [x1, y1, x2, y2] в пикселях благодаря post_process
            box = box.tolist() 
            label_name = dataset.id2label.get(label.item(), str(label.item()))
            
            # Matplotlib Rectangle принимает (x, y, width, height)
            rect = patches.Rectangle(
                (box[0], box[1]), 
                box[2] - box[0], 
                box[3] - box[1], 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(box[0], box[1], f"{label_name}: {score:.2f}", 
                    color='white', fontsize=9, bbox=dict(facecolor='red', alpha=0.5))

        # 2. Рисуем GROUND TRUTH (ЗЕЛЕНЫЕ)
        # Учитываем, что метки могут называться 'class_labels' или 'labels' в зависимости от реализации Datasets
        gt_labels_key = 'class_labels' if 'class_labels' in target else 'labels'
        
        if 'boxes' in target and gt_labels_key in target:
            gt_boxes = target['boxes'] # Ожидаем формат (cx, cy, w, h) нормализованный (0-1)
            gt_classes = target[gt_labels_key]

            for box, label in zip(gt_boxes, gt_classes):
                # Конвертируем нормализованные (cx, cy, w, h) -> пиксельные координаты углов
                cx, cy, w_norm, h_norm = box.tolist()
                
                # Переводим из 0..1 в 0..img_size
                pixel_w = w_norm * img_w
                pixel_h = h_norm * img_h
                pixel_cx = cx * img_w
                pixel_cy = cy * img_h
                
                # Находим левый верхний угол (x, y) для matplotlib
                x_gt = pixel_cx - (pixel_w / 2)
                y_gt = pixel_cy - (pixel_h / 2)
                
                label_name = dataset.id2label.get(label.item(), str(label.item()))

                # Рисуем пунктиром, чтобы отличить от предсказаний
                rect_gt = patches.Rectangle(
                    (x_gt, y_gt), 
                    pixel_w, 
                    pixel_h, 
                    linewidth=2, edgecolor='g', facecolor='none', linestyle='--'
                )
                ax.add_patch(rect_gt)
                # Текст смещаем чуть ниже или в сторону, чтобы не перекрывал красный
                ax.text(x_gt, y_gt - 5, f"GT: {label_name}", 
                        color='white', fontsize=9, bbox=dict(facecolor='green', alpha=0.5))

        plt.title(f"Sample {i}: Red=Pred, Green(dashed)=GT")
        plt.axis('off')
        save_path = f"./runs/analysis_{i}.png"
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close()
        
    print(f"Analysis complete.")


def main():
    # Setup data reuse
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    val_dataset = CocoSubset(
        root=VAL_IMG_DIR, 
        annFile=VAL_ANN, 
        processor=processor, 
        target_classes=TARGET_CLASSES,
        train=False
    )
    
    # Load Model
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=len(TARGET_CLASSES),
        id2label=val_dataset.id2label,
        label2id=val_dataset.label2id,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    
    analyze_predictions(model, processor, val_dataset)

if __name__ == "__main__":
    main()
