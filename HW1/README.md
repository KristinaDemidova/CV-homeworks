# HW1

***

### 1. Подготовка данных

- В качестве датасета выбран Vehicle Image Classification (https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification) с изображениями транспортных средств.
- Данные разбиты на обучающую и валидационную выборки в формате `ImageFolder`.
- Для обучения применены базовые аугментации: случайное обрезание с масштабированием (`RandomResizedCrop`), случайное горизонтальное отражение (`RandomHorizontalFlip`), нормализация.
- Для валидации — базовое изменение размера и центрированное обрезание с нормализацией.
***

### 2. Тренировочный цикл (CNN)

- Построена простая CNN с 2–3 сверточными слоями и классификатором.
- Зафиксированы сиды для воспроизводимости.
- Проведен sanity-check — модель успешно overfit на нескольких батчах.
- Логирование процесса обучения ведется в TensorBoard, включая loss, accuracy, learning rate, гистограммы весов и градиентов.

***

### 3. ViT-Tiny (linear probe)

- Загружен предобученный ViT-Tiny.
- Все слои бэкона заморожены, обучается только линейная классификаторная голова.

***

### 4. Профилировка и логирование

- Использован `torch.profiler` для сбора trace на разном количестве эпох
- Замерены время шага и использование памяти во время обучения.
- Результаты профилировки:
CNN
![cnn_trace.png](/CV-homeworks/HW1/result/cnn_trace.png)

Vit-Tiny

![vit_trace.png](/CV-homeworks/HW1/result/vit_trace.png)

CNN потребляет больше ресурсов GPU

***

### 5. Сравнение моделей

- Метрики качества (accuracy и macro-F1):

![model_comparison.csv](/CV-homeworks/HW1/result/model_comparison.csv)

CNN
![img_2.png](/CV-homeworks/HW1/result/cnn_confusion_matrix.png)

Vit
![img_1.png](/CV-homeworks/HW1/result/vit_confusion_matrix.png)

Accuracy
![img.png](/CV-homeworks/HW1/result/accuracy.jpg)

- Построено и проанализировано confusion matrix для обеих моделей.
- Ключевые выводы:
  - CNN сходится, но при этом потребляет больше ресурсов, чем ViT, хотя ViT требовалось больше времени на эпоху, то есть он упирается не в мощность GPU.
  - ViT требует горздо меньше времени на обучение, выдает высокую точность почти сразу.
***