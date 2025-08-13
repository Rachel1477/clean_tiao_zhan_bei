import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import os
import time
import json
import math
import sys
sys.path.append(os.path.dirname(__file__))  
# 从同级目录导入
from preprocess import SequentialMultimodalRadarDataset, get_vit_transforms
from model import MultimodalTransformerWithLSTM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #当前文件的路径

# 学习率调度器
class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr_scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * lr_scale for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        return [self.eta_min + (base_lr - self.eta_min) * cosine_decay for base_lr in self.base_lrs]


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, writer, grad_clip_value):
    """训练一个轮次"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} Training", leave=False)
    for i, batch in enumerate(progress_bar):
        # 加载批次数据
        rd_map_sequences = batch['rd_map_sequence'].to(device)  # (B, T, C, H, W)
        tabular_sequences = batch['tabular_sequence'].to(device)  # (B, T, F)
        lengths = batch['length'].to(device)  # (B,)
        labels = batch['label'].to(device)

        # 前向传播
        outputs = model(rd_map_sequences, tabular_sequences, lengths)
        loss = criterion(outputs, labels)

        # 检查loss是否为nan
        if torch.isnan(loss):
            print(f"检测到 NaN loss 在 step {i}。停止训练。")
            return float('nan'), float('nan')

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
        optimizer.step()
        scheduler.step()

        # 记录指标
        running_loss += loss.item() * rd_map_sequences.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

        if writer:
            writer.add_scalar('Loss/train_step', loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('LR/step', scheduler.get_last_lr()[0], epoch * len(dataloader) + i)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()


def validate_one_epoch(model, dataloader, criterion, device):
    """验证一个轮次"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        for batch in progress_bar:
            rd_map_sequences = batch['rd_map_sequence'].to(device)
            tabular_sequences = batch['tabular_sequence'].to(device)
            lengths = batch['length'].to(device)
            labels = batch['label'].to(device)

            outputs = model(rd_map_sequences, tabular_sequences, lengths)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * rd_map_sequences.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()


def main(CONFIG):
    # 配置参数
    os.makedirs(CONFIG['model_save_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=CONFIG['log_dir'])
    print(f"使用设备: {device}")

    # 准备数据集和DataLoader
    print("准备数据集中...")
    data_transforms = get_vit_transforms(img_size=CONFIG['img_size'])
    scaler_save_path = os.path.join(CONFIG['model_save_dir'], 'tabular_scaler.npz')
    # 训练集
    train_dataset = SequentialMultimodalRadarDataset(
        data_root_dir=CONFIG['data_root_dir'],
        data_split='train',
        transform=data_transforms,
        max_seq_len=CONFIG['max_seq_len'],
        scaler_path=scaler_save_path,
        mod=CONFIG['mod'],

    )

    # 验证集（使用训练集的scaler）
    val_dataset = SequentialMultimodalRadarDataset(
        data_root_dir=CONFIG['data_root_dir'],
        data_split='val',
        transform=data_transforms,
        max_seq_len=CONFIG['max_seq_len'],
        scaler_path=scaler_save_path,
        mod=CONFIG['mod'],

    )
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 保存标签映射
    label_map_path = os.path.join(CONFIG['model_save_dir'], 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump(train_dataset.label_map, f)
    print(f"标签映射已保存到: {label_map_path}")

    # 初始化模型、损失函数和优化器
    num_classes = len(train_dataset.label_map)
    # 计算表格特征数量（点迹+航迹+时间间隔）
    num_tabular_features = len(train_dataset.point_cols) + len(train_dataset.track_cols)

    model = MultimodalTransformerWithLSTM(
        img_size=CONFIG['img_size'][0],
        patch_size=32,
        in_channels=1,
        num_tabular_features=num_tabular_features,
        num_classes=num_classes,
        embed_dim=128,
        depth=3,
        heads=4,
        mlp_dim=256,
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)

    # 初始化学习率调度器
    total_steps = len(train_loader) * CONFIG['num_epochs']
    warmup_steps = len(train_loader) * CONFIG['warmup_epochs']
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    # 训练循环
    best_val_acc = 0.0
    epochs_no_improve = 0
    print("\n--- 开始训练 (Transformer + LSTM) ---")

    for epoch in range(CONFIG['num_epochs']):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, writer, CONFIG['grad_clip']
        )

        # 检查训练是否因nan loss中断
        if math.isnan(train_loss):
            print("训练因 NaN loss 终止。")
            break

        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
              f"Duration: {epoch_duration:.2f}s")

        # 记录到TensorBoard
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', train_acc, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        writer.add_scalar('Accuracy/val_epoch', val_acc, epoch)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            model_path = os.path.join(CONFIG['model_save_dir'], 'best.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  -> 验证准确率提升，保存模型到 {model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG['patience']:
                print(f"  -> 连续 {CONFIG['patience']} 个epoch验证准确率没有改善，提前停止训练。")
                break

    writer.close()
    print("\n--- 训练完成 ---")


if __name__ == '__main__':

    CONFIG = {
        'data_root_dir': os.path.join(BASE_DIR, '../'),
        'img_size': (224, 224),
        'batch_size': 16,  # 时序模型通常需要更小的批次
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'patience': 10,
        'model_save_dir': os.path.join(BASE_DIR, 'model_save'),
        'log_dir': os.path.join(BASE_DIR, 'logs'),
        'grad_clip': 1.0,
        'warmup_epochs': 2,
        'max_seq_len': 30,  # 最大序列长度，根据数据分布设置
        'mod':'train'
    }

    main(CONFIG=CONFIG) 