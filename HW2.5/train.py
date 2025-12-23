import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset
import time
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def train_model(train_datasets, val_dir, num_classes, num_epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Трансформации
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Подготовка данных
    # Если train_datasets это список (real + synthetic), объединяем их
    if isinstance(train_datasets, list):
        # Применяем трансформ к каждому и объединяем
        transformed_list = []
        for d_path in train_datasets:
            d = datasets.ImageFolder(d_path, transform=data_transforms)
            transformed_list.append(d)
        train_data = ConcatDataset(transformed_list)
        # Важно: проверить маппинг классов, если папки синтетики совпадают с реальными
    else:
        train_data = datasets.ImageFolder(train_datasets, transform=data_transforms)
        
    val_data = datasets.ImageFolder(val_dir, transform=data_transforms)
    
    dataloaders = {
        'train': DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    }
    
    # Модель (ResNet18 для примера)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    history = {'train_acc': [], 'val_acc': []}

    print(f"Начинаем обучение на устройстве: {device}")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            history[f'{phase}_acc'].append(epoch_acc.item())
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                
    return best_acc, history
