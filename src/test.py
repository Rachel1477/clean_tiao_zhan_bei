import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from PIL import Image
import re
import sys

# 假设 model.py, preprocess.py, utils.py 都在同级目录
try:
    from preprocess import get_vit_transforms
    from model import MultimodalTransformerWithLSTM
    from utils import parse_time_to_seconds, clean_tabular_dataframe, calculate_time_features
except ImportError:
    print("错误: 无法导入必要的模块。请确保所有.py文件都在同一目录。")
    sys.exit(1)

def load_model(model_path, device, model_class, **model_args):
    """加载训练好的模型。"""
    print(f"正在从 '{model_path}' 加载模型...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    model = model_class(**model_args).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        raise e
    model.eval()
    print("模型加载成功！")
    return model

def main():
    # --- 1. 配置参数：必须与训练时使用的 train.py 中的 CONFIG 一致！ ---
    CONFIG = {
        'test_data_root': '../',
        'model_dir': './model_save', # ★★★ 修正：指向正确的模型目录
        'model_filename': 'best.pth', # ★★★ 修正：指向正确的模型文件名
        'output_dir': './test_results_output',
        'img_size': (224, 224),
        
        # --- ★★★ 核心修正：模型超参数必须和训练时完全一样 ★★★ ---
        'patch_size': 32, # 根据错误报告，训练时使用的是32
        'embed_dim': 128,
        'depth': 3,
        'heads': 4,
        'mlp_dim': 256,
        'lstm_hidden_dim': 128,
        'lstm_num_layers': 2,
        'dropout': 0.1,
    }

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 2. 加载元数据 ---
    print("加载模型工件 (标签映射, 标准化参数)...")
    label_map_path = os.path.join(CONFIG['model_dir'], 'label_map.json')
    scaler_path = os.path.join(CONFIG['model_dir'], 'tabular_scaler.npz')
    
    try:
        with open(label_map_path, 'r') as f: label_map = json.load(f)
        inv_label_map = {v: k for k, v in label_map.items()}
        num_classes = len(label_map)
        
        scaler_data = np.load(scaler_path)
        tabular_scaler = {'mean': scaler_data['mean'], 'std': scaler_data['std']}
        print("成功加载标签映射和标准化参数。")
    except FileNotFoundError as e:
        print(f"错误: 找不到必要的工件文件: {e}")
        return

    # ★★★ 核心修正：表格特征定义必须和训练时完全一样 ★★★
    point_cols = ['距离', '方位', '俯仰', '多普勒速度', '和幅度', '信噪比', '原始点数量']
    track_cols = ['滤波距离', '滤波方位', '滤波俯仰', '全速度', 'X向速度', 'Y向速度', 'Z向速度', '航向']
    # 根据错误报告，训练时的总维度是16，所以这里是 7 + 8 + 1(时间) = 16
    num_tabular_features_with_time = len(point_cols) + len(track_cols) + 1 
    all_tabular_cols_with_time = point_cols + track_cols + ['delta_t']

    # --- 3. 加载模型 ---
    model_path = os.path.join(CONFIG['model_dir'], CONFIG['model_filename'])
    # 从CONFIG中动态构建模型参数字典
    model_args = {
        'img_size': CONFIG['img_size'][0],
        'patch_size': CONFIG['patch_size'],
        'in_channels': 1,
        'num_tabular_features': num_tabular_features_with_time - 1, # 模型__init__接收的是不含时间的
        'num_classes': num_classes,
        'embed_dim': CONFIG['embed_dim'],
        'depth': CONFIG['depth'],
        'heads': CONFIG['heads'],
        'mlp_dim': CONFIG['mlp_dim'],
        'lstm_hidden_dim': CONFIG['lstm_hidden_dim'],
        'lstm_num_layers': CONFIG['lstm_num_layers'],
        'dropout': CONFIG['dropout']
    }
    
    model = load_model(model_path=model_path, device=device, model_class=MultimodalTransformerWithLSTM, **model_args)

    # --- 4. 发现并准备测试数据 ---
    data_transforms = get_vit_transforms(img_size=CONFIG['img_size'])
    track_dir = os.path.join(CONFIG['test_data_root'], '航迹')
    point_dir = os.path.join(CONFIG['test_data_root'], '点迹')
    npy_base_dir = os.path.join(CONFIG['test_data_root'], 'dataset')
    
    track_info_pattern = re.compile(r'Tracks_(\d+)_(\d+)\.txt')
    test_tracks = [{'id': m.group(1), 'len': m.group(2)} for f in os.listdir(track_dir) if (m := track_info_pattern.match(f))]
    
    if not test_tracks:
        print(f"错误: 在 '{track_dir}' 目录中未发现任何测试航迹文件。")
        return
    print(f"\n发现 {len(test_tracks)} 个待测试航迹。")

    # --- 5. 在线预测主循环 ---
    with torch.no_grad():
        for track_info in tqdm(test_tracks, desc="处理测试航迹"):
            track_id, track_len = track_info['id'], track_info['len']
            
            try:
                point_df = pd.read_csv(os.path.join(point_dir, f'PointTracks_{track_id}_{track_len}.txt'), encoding='GBK')
                track_df_original = pd.read_csv(os.path.join(track_dir, f'Tracks_{track_id}_{track_len}.txt'), encoding='GBK')
                
                point_df['time_seconds'] = point_df[point_df.columns[0]].apply(parse_time_to_seconds)
                point_df_cleaned = clean_tabular_dataframe(point_df, point_cols + ['time_seconds'])
                point_df_featured = calculate_time_features(point_df_cleaned, time_col='time_seconds')
                track_df_cleaned = clean_tabular_dataframe(track_df_original.copy(), track_cols)
            except Exception as e:
                print(f"\n加载或处理航迹 {track_id} 的表格数据时出错: {e}，跳过。")
                continue

            hidden_state, predicted_label_sequence = None, []
            
            for i in range(len(track_df_original)):
                try:
                    tabular_features_raw = np.concatenate([
                        point_df_featured.iloc[i][point_cols].values,
                        track_df_cleaned.iloc[i][track_cols].values,
                        [point_df_featured.iloc[i]['delta_t']]
                    ]).astype(np.float32)

                    tabular_features_scaled = (tabular_features_raw - tabular_scaler['mean']) / (tabular_scaler['std'] + 1e-8)
                    tabular_features_t = torch.tensor(tabular_features_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # 构造NPY文件路径
                    npy_folder_path = os.path.join(npy_base_dir, track_id)
                    npy_filename = f"{track_id}_Point_{i+1}.npy" 
                    rd_map_path = os.path.join(npy_folder_path+"_" ,npy_filename)
                    
                    if not os.path.exists(rd_map_path):
                        last_pred = predicted_label_sequence[-1] if predicted_label_sequence else '1'
                        predicted_label_sequence.append(last_pred)
                        continue
                        
                    rd_map_raw = np.load(rd_map_path)
                    rd_map_normalized = (rd_map_raw - np.min(rd_map_raw)) / (np.max(rd_map_raw) - np.min(rd_map_raw) + 1e-8)
                    image = Image.fromarray((rd_map_normalized * 255).astype(np.uint8))
                    rd_map_t = data_transforms(image).unsqueeze(0).to(device)

                    # 在线预测
                    logits, hidden_state = model.forward_online(rd_map_t, tabular_features_t, hidden_state)
                    _, pred = torch.max(logits, 1)
                    predicted_label_sequence.append(inv_label_map[pred.item()])
                except Exception as e:
                    print("xxxxxxxxxx",e)
                    last_pred = predicted_label_sequence[-1] if predicted_label_sequence else '1'
                    predicted_label_sequence.append(last_pred)

            if predicted_label_sequence:
                output_df = track_df_original.copy()
                output_df['识别结果'] = predicted_label_sequence
                output_filepath = os.path.join(CONFIG['output_dir'], f'Tracks_{track_id}_{track_len}.txt')
                output_df.to_csv(output_filepath, index=False, encoding='GBK')

    print(f"\n--- 预测完成 ---")
    print(f"所有结果文件已保存在: {CONFIG['output_dir']}")

if __name__ == '__main__':
    main()