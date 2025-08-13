import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import sys
sys.path.append(os.path.dirname(__file__))  

from preprocess import SequentialMultimodalRadarDataset, get_vit_transforms
from model import MultimodalTransformerWithLSTM
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

def load_model(model_path, device, model_class, **model_args):
    """加载训练好的时序多模态模型"""
    print(f"正在从 '{model_path}' 加载模型...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    model = model_class(**model_args).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"加载模型权重时出错: {e}")
        print("这可能是因为模型定义与保存的权重不匹配。请检查模型参数。")
        raise e
    model.eval()  # 评估模式
    print("模型加载成功！")
    return model

def plot_confusion_matrix(cm, class_names, output_path):
    """绘制并保存混淆矩阵图"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Track-Level Final Decision)', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"混淆矩阵图已保存到: {output_path}")


def main(CONFIG):

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载标签映射和标准化参数...")
    label_map_path = os.path.join(CONFIG['model_dir'], 'label_map.json')
    if not os.path.exists(label_map_path):
        print(f"错误: 找不到标签映射文件 {label_map_path}。请先成功运行train.py。")
        return
        
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [inv_label_map[i] for i in range(len(label_map))]
    num_classes = len(label_map)

    print("临时加载训练集以获取tabular_scaler和列定义...")
    scaler_save_path = os.path.join(CONFIG['model_dir'], 'tabular_scaler.npz')
    temp_dataset = SequentialMultimodalRadarDataset(data_root_dir=CONFIG['data_root_dir'], data_split='test', max_seq_len=30,mod=CONFIG['mod'],scaler_path=scaler_save_path)
    tabular_scaler = temp_dataset.tabular_scaler
    point_cols = temp_dataset.point_cols
    track_cols = temp_dataset.track_cols
    num_tabular_features = len(point_cols) + len(track_cols) 

    model_path = os.path.join(CONFIG['model_dir'], CONFIG['model_filename'])
    model = load_model(
        model_path=model_path, device=device, model_class=MultimodalTransformerWithLSTM,
        img_size=CONFIG['img_size'][0],
        patch_size=CONFIG['patch_size'],
        in_channels=1,
        num_tabular_features=num_tabular_features,
        num_classes=num_classes,
        embed_dim=CONFIG['embed_dim'],
        depth=CONFIG['depth'],
        heads=CONFIG['heads'],
        mlp_dim=CONFIG['mlp_dim'],
        lstm_hidden_dim=CONFIG['lstm_hidden_dim'],
        lstm_num_layers=CONFIG['lstm_num_layers'],
        dropout=CONFIG['dropout']
    )

    val_dataset = SequentialMultimodalRadarDataset(data_root_dir=CONFIG['data_root_dir'], data_split=CONFIG['val_data_dir_name'], max_seq_len=1000,mod=CONFIG['mod'],scaler_path=scaler_save_path) # max_seq_len设大一点确保加载所有点
    val_dataset.tabular_scaler = tabular_scaler
    data_transforms = get_vit_transforms(img_size=CONFIG['img_size'])
    
    print("\n--- 开始在线模拟预测 (逐点更新) ---")
    
    all_results_for_csv = []
    final_true_labels = []
    final_pred_labels = []
    effective_points = []

    with torch.no_grad():
        for track_data in tqdm(val_dataset.tracks, desc="逐航迹处理"):
            true_label_str = track_data['label']
            true_label = label_map[true_label_str]
            track_id = track_data['track_id']
            
            hidden_state = None
            predicted_label_sequence = []

            for t, step_data in enumerate(track_data['steps']):
                rd_map_raw = step_data['rd_map_raw']
                rd_map_normalized = (np.clip(rd_map_raw, np.min(rd_map_raw), np.max(rd_map_raw)) - np.min(rd_map_raw)) / (np.max(rd_map_raw) - np.min(rd_map_raw) + 1e-8)
                rd_map_uint8 = (rd_map_normalized * 255).astype(np.uint8)
                image = Image.fromarray(rd_map_uint8)
                rd_map_t = data_transforms(image).unsqueeze(0).to(device)

                tabular_features_raw = step_data['tabular_features']
                mean = tabular_scaler['mean']
                std = tabular_scaler['std']
                tabular_features_scaled = (tabular_features_raw - mean) / (std + 1e-8)
                tabular_features_t = torch.tensor(tabular_features_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                
                logits, hidden_state = model.forward_online(rd_map_t, tabular_features_t, hidden_state)
                
                _, pred = torch.max(logits, 1)
                current_prediction = pred.item()
                predicted_label_sequence.append(current_prediction)

                all_results_for_csv.append({
                    'Track_ID': track_id,
                    'Point_ID': t + 1,
                    'Predicted_Label': inv_label_map[current_prediction],
                    'True_Label': true_label_str
                })

            if not predicted_label_sequence: continue

            all_results_for_csv.append({'Track_ID': 'end', 'Point_ID': 'end', 'Predicted_Label': 'end', 'True_Label': 'end'})
            final_true_labels.append(true_label)
            final_pred_labels.append(predicted_label_sequence[-1])

    print("\n--- 在线模拟评估结果 ---")
    
    report = classification_report(final_true_labels, final_pred_labels, target_names=class_names, digits=4, zero_division=0)
    print("分类报告 (基于航迹最终决策):")
    print(report)

    cm = confusion_matrix(final_true_labels, final_pred_labels, labels=list(range(num_classes)))
    cm_output_path = os.path.join(CONFIG['output_dir'], 'online_confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_output_path)
    
    results_df = pd.DataFrame(all_results_for_csv)
    csv_output_path = os.path.join(CONFIG['output_dir'], 'online_prediction_details.csv')
    results_df.to_csv(csv_output_path, index=False)
    print(f"详细预测结果已保存到: {csv_output_path}")


if __name__ == '__main__':
    CONFIG = {
        'data_root_dir': os.path.join(BASE_DIR, '../'),
        'val_data_dir_name': 'test',
        'model_dir': os.path.join(BASE_DIR, 'model_save'),
        'model_filename': 'best.pth',
        'output_dir': os.path.join(BASE_DIR,'online_prediction_results'),
        'img_size': (224, 224),
        'patch_size': 32, 
        'embed_dim': 128,
        'depth': 3,
        'heads': 4,
        'mlp_dim': 256,
        'lstm_hidden_dim': 128,
        'lstm_num_layers': 2,
        'dropout': 0.1,
        'mod':'test',
    }
    main(CONFIG)