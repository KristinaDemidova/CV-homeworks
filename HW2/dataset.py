import torch
import os
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from PIL import Image

class CocoSubset(Dataset):
    def __init__(self, root, annFile, processor, target_classes=None, train=True):
        self.coco_dataset = CocoDetection(root=root, annFile=annFile)
        self.processor = processor
        self.target_classes = target_classes
        
        # Загружаем категории
        cats = self.coco_dataset.coco.loadCats(self.coco_dataset.coco.getCatIds())
        self.all_cat_ids = {cat['id']: cat['name'] for cat in cats}
        
        if target_classes:
            self.cat_ids_to_keep = self.coco_dataset.coco.getCatIds(catNms=target_classes)
            self.label2id = {label: i for i, label in enumerate(target_classes)}
            self.id2label = {i: label for i, label in enumerate(target_classes)}
            # Маппинг из оригинального ID COCO в наш ID (от 0 до 9)
            self.coco_id_to_internal = {
                coco_id: self.label2id[self.all_cat_ids[coco_id]] 
                for coco_id in self.cat_ids_to_keep
            }
            
            # Фильтруем картинки, оставляем только те, где есть наши классы
            self.ids = []
            for img_id in self.coco_dataset.ids:
                ann_ids = self.coco_dataset.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids_to_keep)
                if len(ann_ids) > 0:
                    self.ids.append(img_id)
        else:
            self.ids = self.coco_dataset.ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco_dataset.coco.loadImgs(img_id)[0]
        
        # Получаем аннотации
        ann_ids = self.coco_dataset.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids_to_keep if self.target_classes else None)
        anns = self.coco_dataset.coco.loadAnns(ann_ids)
        
        # Загружаем картинку
        image_path = os.path.join(self.coco_dataset.root, img_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            # Игнорируем слишком мелкие или "сломанные" боксы
            if w < 1 or h < 1: continue
            
            current_coco_id = ann['category_id']
            # Если у нас подмножество классов, проверяем и конвертируем ID
            if self.target_classes:
                if current_coco_id in self.coco_id_to_internal:
                    boxes.append(ann['bbox']) 
                    labels.append(self.coco_id_to_internal[current_coco_id])
                    areas.append(ann['area'])
                    iscrowd.append(ann['iscrowd'])
            else:
                # Если обучаем на всем, берем как есть (но тогда target_classes должен быть None)
                boxes.append(ann['bbox'])
                labels.append(current_coco_id) # Тут может потребоваться ремаппинг, если ID в COCO дырявые

        # Готовим структуру для процессора
        encoding = self.processor(
            images=image, 
            annotations={'image_id': img_id, 'annotations': [
                {'bbox': b, 'category_id': l, 'area': a, 'iscrowd': ic} 
                for b, l, a, ic in zip(boxes, labels, areas, iscrowd)
            ]}, 
            return_tensors="pt"
        )
        
        # Убираем лишнюю размерность батча (так как процессор добавляет batch dimension = 1)
        pixel_values = encoding["pixel_values"].squeeze() 
        target = encoding["labels"][0] 

        return pixel_values, target

class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        # Разбираем батч на картинки и метки
        pixel_values = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
        # Передаем список тензоров (pixel_values) напрямую в pad.
        # Процессор сам создаст pixel_mask и выровняет размеры.
        batch_padded = self.processor.pad(pixel_values, return_tensors="pt")
        
        return {
            'pixel_values': batch_padded['pixel_values'],
            'pixel_mask': batch_padded['pixel_mask'],
            'labels': labels
        }
