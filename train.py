import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from datetime import datetime
import copy
import glob
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# 설정
class Config:
    # 데이터 경로 (본인 환경에 맞게 수정)
    base_dir = './food_data'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    
    # 모델
    model_type = 'mobilenet'
    
    # 학습 설정
    num_classes = 150
    batch_size = 64
    num_epochs = 30
    learning_rate = 0.01
    momentum = 0.9
    img_size = 224
    
    # 체크포인트 설정 ( 강제 종료 후 재 시작시 이어서 학습 가능 )
    checkpoint_dir = f'./checkpoints/{model_type}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_every_n_epochs = 1  # 1 에포크마다 저장
    resume_from = None  # 재개할 체크포인트 경로 (None이면 처음부터)
    
    # 디바이스
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 데이터 로더
def get_data_loaders(config):
    """데이터 로더 생성"""
    
    # Train: Data Augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Val/Test: Augmentation 없음
    val_transforms = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(config.train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(config.val_dir, transform=val_transforms)
    test_dataset = datasets.ImageFolder(config.test_dir, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}')
    print(f'Classes: {len(train_dataset.classes)}')
    
    return train_loader, val_loader, test_loader, train_dataset.classes


# 모델 생성
def create_model(config): 
    model = models.mobilenet_v2(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
         nn.Linear(num_features, config.num_classes)
    )
    
    return model.to(config.device)


# 체크포인트 저장/로드
def save_checkpoint(epoch, model, optimizer, scheduler, best_val_acc, 
                   class_names, config, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'class_names': class_names,
        'config': vars(config)
    }
    
    # 일반 체크포인트
    checkpoint_path = os.path.join(
        config.checkpoint_dir, 
        f'checkpoint_epoch_{epoch:03d}.pth'
    )
    torch.save(checkpoint, checkpoint_path)
    
    # 최고 성능 모델 별도 저장
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f'✓ Best model saved (epoch {epoch}, val_acc: {best_val_acc:.4f})')
    
    # 최신 체크포인트도 별도 저장 (재개용)
    latest_path = os.path.join(config.checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path

# 체크포인트 로드 및 재개
def load_checkpoint(checkpoint_path, model, optimizer, scheduler, config):
    print(f'\n체크포인트 로드 중: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc']
    
    print(f'재개: Epoch {start_epoch}부터 시작 (Best Val Acc: {best_val_acc:.4f})')
    
    return start_epoch, best_val_acc

# 가장 최근 체크포인트 찾기
def find_latest_checkpoint(config):
    latest_path = os.path.join(config.checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(latest_path):
        return latest_path
    
    # latest가 없으면 에포크 번호가 가장 큰 것 찾기
    checkpoints = glob.glob(os.path.join(config.checkpoint_dir, 'checkpoint_epoch_*.pth'))
    if checkpoints:
        return max(checkpoints, key=os.path.getctime)
    
    return None


# 학습률 자동 조정 스케쥴러 
def get_lr_scheduler(optimizer, config):
    def lr_lambda(epoch):
        if epoch < 15:
            return 1.0
        elif epoch < 28:
            return 0.2
        else:
            return 0.04
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# 학습/검증 함수 ( 하나의 에포크의 학습 및 검증 )
def train_one_epoch(model, train_loader, criterion, optimizer, config):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # tqdm 진행률 표시
    pbar = tqdm(train_loader, desc='Training', leave=False)
    
    for inputs, labels in pbar:
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)
        
        optimizer.zero_grad()
        
        # Forward
        if config.model_type == 'inception' and model.training:
            outputs, aux_outputs = model(inputs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        # 실시간 loss/acc 표시
        current_loss = running_loss / total_samples
        current_acc = running_corrects.double() / total_samples
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.4f}'
        })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc

# 검증
def validate(model, val_loader, criterion, config):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # tqdm 진행률 표시
    pbar = tqdm(val_loader, desc='Validation', leave=False)
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # 실시간 loss/acc 표시
            current_loss = running_loss / total_samples
            current_acc = running_corrects.double() / total_samples
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc


# 메인 학습 루프
def train_model(config):
    print('\n' + '='*70)
    print('한식 이미지 분류 모델 학습 시작')
    print('='*70)
    print(f'  Batch size: {config.batch_size}')
    print(f'  Epochs: {config.num_epochs}')
    print(f'  Checkpoint dir: {config.checkpoint_dir}')
    print('='*70 + '\n')
    
    # 데이터 로더
    train_loader, val_loader, test_loader, class_names = get_data_loaders(config)
    
    # 모델, 손실함수, 옵티마이저
    model = create_model(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    scheduler = get_lr_scheduler(optimizer, config)
    
    # 체크포인트에서 재개
    start_epoch = 0
    best_val_acc = 0.0
    
    if config.resume_from:
        start_epoch, best_val_acc = load_checkpoint(
            config.resume_from, model, optimizer, scheduler, config
        )
    elif config.resume_from is None:
        # 자동으로 latest checkpoint 찾기
        latest_checkpoint = find_latest_checkpoint(config)
        if latest_checkpoint:
            print(f'\n최근 체크포인트 발견: {latest_checkpoint}')
            resume = input('이어서 시작하시겠습니까? (y/n): ').strip().lower()
            if resume == 'y':
                start_epoch, best_val_acc = load_checkpoint(
                    latest_checkpoint, model, optimizer, scheduler, config
                )
    
    # Early Stopping
    patience = 10
    patience_counter = 0
    
    print('\n' + '='*70)
    print('학습 시작중')
    print('='*70)
    
    for epoch in range(start_epoch, config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        print('-' * 70)
        
        # 학습
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config)
        
        # 검증
        val_loss, val_acc = validate(model, val_loader, criterion, config)
        
        # Learning rate 업데이트
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 출력
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}')
        print(f'LR: {current_lr:.6f}')
        
        # 체크포인트 저장
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 정기적으로 체크포인트 저장
        if (epoch + 1) % config.save_every_n_epochs == 0 or is_best:
            save_checkpoint(
                epoch, model, optimizer, scheduler, best_val_acc,
                class_names, config, is_best=is_best
            )
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break
    
    # 최종 평가 (Test set)
    print('\n' + '='*70)
    print('테스트셋으로 최종 평가 시작중')
    print('='*70)
    
    # Best model 로드
    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'✓ Best model loaded (Val Acc: {checkpoint["best_val_acc"]:.4f})')
    
    test_loss, test_acc = validate(model, test_loader, criterion, config)
    print(f'\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')
    
    print('\n✓ Training completed!')
    print(f'✓ Best model saved at: {best_model_path}')
    
    return model

# 메인 실행 made in claude
if __name__ == '__main__':
    config = Config()

    model = train_model(config)