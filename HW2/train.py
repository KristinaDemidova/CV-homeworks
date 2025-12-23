import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import torch.profiler
import os
# --- ИЗМЕНЕНИЕ 1: Импортируем Collator вместо collate_fn ---
from dataset import CocoSubset, Collator 

# --- HYPERPARAMETERS ---
BATCH_SIZE = 4
LR = 1e-5
LR_BACKBONE = 1e-5
EPOCHS = 10
NUM_WORKERS = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_DIR = "./runs/detr_experiment"
CHECKPOINT_DIR = "./checkpoints"

# Paths (Если вы используете download_coco.py, пути такие)
TRAIN_IMG_DIR = "data/train/data"
TRAIN_ANN = "data/train/labels.json"

TARGET_CLASSES = [
    'person', 'car', 'bicycle', 'bus', 'truck', 
    'traffic light', 'stop sign', 'cat', 'dog', 'chair'
]

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 1. Setup Data
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    train_dataset = CocoSubset(
        root=TRAIN_IMG_DIR, 
        annFile=TRAIN_ANN, 
        processor=processor, 
        target_classes=TARGET_CLASSES
    )
    
    # --- ИЗМЕНЕНИЕ 2: Инициализируем коллатор ---
    data_collator = Collator(processor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=data_collator, # Используем наш класс
        shuffle=True, 
        num_workers=NUM_WORKERS
    )

    id2label = train_dataset.id2label
    label2id = train_dataset.label2id

    # 2. Setup Model
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=len(TARGET_CLASSES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)

    # 3. Optimizer
    #param_dicts = [
    #    {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    #    {
    #        "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
    #        "lr": LR_BACKBONE,
    #    },
    #]
    param_dicts = [
    {
        "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    },
]
    optimizer = AdamW(param_dicts, lr=LR, weight_decay=1e-4)
    
    # 4. Logging & Profiler
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    print(f"Starting training on {DEVICE}...")
    global_step = 0
    
    # Можно отключить профайлер, если будут ошибки памяти, или убрать schedule
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(LOG_DIR),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
    
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            
            for step, batch in enumerate(train_loader):
                pixel_values = batch['pixel_values'].to(DEVICE)
                # --- ИЗМЕНЕНИЕ 3: Берем pixel_mask ---
                pixel_mask = batch['pixel_mask'].to(DEVICE)
                
                labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch['labels']]
                
                optimizer.zero_grad()
                
                # --- ИЗМЕНЕНИЕ 4: Передаем pixel_mask в модель ---
                outputs = model(
                    pixel_values=pixel_values, 
                    pixel_mask=pixel_mask, 
                    labels=labels
                )
                
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                try:
                    prof.step()
                except:
                    pass # Иногда профайлер капризничает на последних шагах
                
                current_loss = loss.item()
                epoch_loss += current_loss
                writer.add_scalar("Train/Total_Loss", current_loss, global_step)
                
                if hasattr(outputs, "loss_dict"):
                    for k, v in outputs.loss_dict.items():
                        writer.add_scalar(f"Train/{k}", v.item(), global_step)

                if step % 10 == 0:
                    print(f"Epoch {epoch} | Step {step} | Loss: {current_loss:.4f}")
                
                global_step += 1

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch} finished. Avg Loss: {avg_epoch_loss:.4f}")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, os.path.join(CHECKPOINT_DIR, f"detr_ep{epoch}.pth"))

    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    main()
