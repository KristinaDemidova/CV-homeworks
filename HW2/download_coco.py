import fiftyone as fo
import fiftyone.zoo as foz

# Список классов для вашего задания
target_classes = [
    'person', 'car', 'bicycle', 'bus', 'truck', 
    'traffic light', 'stop sign', 'cat', 'dog', 'chair'
]

print("Скачивание Subset Train...")
# Скачиваем Train (ограничим например 2000-5000 картинок, чтобы было быстро)
# Если хотите весь COCO с этими классами, уберите max_samples
dataset_train = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=target_classes,
    max_samples=2500,  # Закомментируйте, если хотите все картинки этих классов
#    dataset_dir="./data_subset"
)

# Экспортируем в стандартный COCO формат, который понимает наш код
dataset_train.export(
    export_dir="./data/train",
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
)

print("Скачивание Subset Val...")
dataset_val = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=target_classes,
    max_samples=500,
#    dataset_dir="./data_subset"
)

dataset_val.export(
    export_dir="./data/val",
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
)

print("Готово! Данные лежат в папке data/")
