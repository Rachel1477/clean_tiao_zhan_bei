import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import re

err_list=[]
class SequentialMultimodalRadarDataset(Dataset):
    """
    时序多模态雷达数据集类，为Transformer+LSTM模型准备数据。
    按航迹组织数据，每条航迹包含多个时间步。
    """

    def __init__(self, data_root_dir, data_split, mod, scaler_path, transform=None, tabular_cols=None, max_seq_len=30):
        super().__init__()
        self.mod=mod
        self.data_root_dir = data_root_dir
        self.data_split = data_split
        self.transform = transform
        self.max_seq_len = max_seq_len  # 最大序列长度
        self.npy_dir = os.path.join(data_root_dir, 'DRmap', data_split)
        self.point_track_dir = os.path.join(data_root_dir, '点迹')
        self.track_dir = os.path.join(data_root_dir, '航迹')

        # 定义需要抽取的表格数据列
        if tabular_cols is None:
            self.point_cols = ['距离', '方位', '俯仰', '多普勒速度', '和幅度', '信噪比', '原始点数量']
            self.track_cols = ['滤波距离', '滤波方位', '滤波俯仰', '全速度', 'X向速度', 'Y向速度', 'Z向速度', '航向']
        else:
            self.point_cols = tabular_cols.get('point', [])
            self.track_cols = tabular_cols.get('track', [])

        self.tracks = []  # 存储航迹数据，每条航迹是一个时序序列
        self.label_map = {}
        self.tabular_data_cache = {}

        print(f'npy_dir:{self.npy_dir}, point_track_dir{self.point_track_dir}, track_dir:{self.track_dir}')
        if not all(os.path.isdir(d) for d in [self.npy_dir, self.point_track_dir, self.track_dir]):
            raise FileNotFoundError("一个或多个数据目录未找到。请检查路径。")

        print(f"开始为 '{data_split}' 集加载时序多模态数据...")
        self._load_tracks()  # 按航迹加载数据
        self._create_label_map()
        self.tabular_scaler = None
        if mod == 'train':
            self._create_tabular_scaler(scaler_path)
        elif mod == 'val' or 'test':
            self._load_tabular_scaler(scaler_path)
        print("数据加载完成。")
        print(f"共找到 {len(self.tracks)} 条航迹。")
        print(f"类别映射: {self.label_map}")

    def _load_tabular_data(self, track_id, label, track_len):
        """加载一个航迹对应的点迹和航迹txt文件"""
        cache_key = f"{track_id}_{label}"
        if cache_key in self.tabular_data_cache:
            return self.tabular_data_cache[cache_key]
        try:
            point_filename = f'PointTracks_{track_id}_{label}_{track_len}.txt'
            point_filepath = os.path.join(self.point_track_dir, point_filename)
            point_df = pd.read_csv(point_filepath, encoding='GBK')

            track_filename = f'Tracks_{track_id}_{label}_{track_len}.txt'
            track_filepath = os.path.join(self.track_dir, track_filename)
            track_df = pd.read_csv(track_filepath, encoding='GBK')

            # 提取时间列并计算时间间隔
            if '点时间' in point_df.columns:
                # 处理时间格式 'HH:MM:SS.ssssss'
                # 由于没有日期信息，我们将计算相对时间差
                # 先转换为timedelta类型
                point_df['点时间'] = pd.to_timedelta(point_df['点时间'])

                # 计算与第一个点的时间间隔(秒)
                # 这比使用diff()更稳定，因为timedelta可以正确处理跨天的情况
                first_time = point_df['点时间'].iloc[0]
                point_df['时间间隔'] = (point_df['点时间'] - first_time).dt.total_seconds()

                # 确保第一个时间间隔为0
                point_df.loc[0, '时间间隔'] = 0.0

            self.tabular_data_cache[cache_key] = (point_df, track_df)
            return point_df, track_df
        except Exception as e:
            print(f"读取txt文件时出错 {cache_key}: {e}")
            return None, None

    def _load_tracks(self):
        """
        按航迹加载时序数据。
        修改后的逻辑：以Tracks.txt文件为基准进行迭代，确保时序完整性。
        如果某个时间步缺少对应的.npy文件，则为其创建一个零矩阵作为占位符。
        """
        # 正则表达式用于从.npy文件名中解析信息，现在用于构建预期的文件名
        file_pattern_base = "{track_id}_Label_{label}_Point_{point_index}.npy"

        # 遍历所有航迹文件夹
        for track_folder in tqdm(os.listdir(self.npy_dir), desc=f"扫描 {self.data_split} 航迹"):
            current_track_path = os.path.join(self.npy_dir, track_folder)
            if not os.path.isdir(current_track_path):
                continue

            # 解析航迹ID和标签
            track_folder_parts = track_folder.split('_')
            if len(track_folder_parts) < 3:
                print(f"发现不合规的航迹文件夹名: {track_folder}")
                continue
            track_id, label = track_folder_parts[0], track_folder_parts[2]

            # 查找并确认唯一的表格数据文件
            track_txt_files = [f for f in os.listdir(self.track_dir) if f.startswith(f'Tracks_{track_id}_{label}_')]
            if not track_txt_files:
                print(f"警告: 航迹 {track_id} 找不到对应的 'Tracks_{track_id}_{label}_*.txt' 文件，已跳过。")
                continue
            
            # 从文件名中解析航迹长度
            track_len_str = track_txt_files[0].split('_')[-1].replace('.txt', '')

            # 加载表格数据
            point_df, track_df = self._load_tabular_data(track_id, label, track_len_str)
            if point_df is None or track_df is None:
                continue

            # 数据清洗（与之前相同）
            for col in self.point_cols + ['时间间隔']:
                if col in point_df.columns:
                    point_df[col] = pd.to_numeric(point_df[col], errors='coerce')
            point_df.fillna(point_df.mean(numeric_only=True), inplace=True)
            point_df.fillna(0, inplace=True)

            for col in self.track_cols:
                if col in track_df.columns:
                    track_df[col] = pd.to_numeric(track_df[col], errors='coerce')
            track_df.fillna(track_df.mean(numeric_only=True), inplace=True)
            track_df.fillna(0, inplace=True)

            # --- 核心逻辑修改 ---
            # 之前的逻辑是查找所有存在的.npy文件，现在改为以txt文件的行数为准进行迭代
            track_steps = []
            num_steps_in_df = min(len(point_df), len(track_df)) # 以较短的数据帧为准，防止索引越界

            for i in range(num_steps_in_df):
                # 限制最大序列长度
                if i >= self.max_seq_len:
                    break
                try:
                    # 1. 提取表格特征（这部分总会执行）
                    point_features = point_df.iloc[i][self.point_cols].values.astype(np.float32)
                    time_interval = point_df.iloc[i]['时间间隔'] if '时间间隔' in point_df.columns else 0
                    time_interval = np.array([time_interval], dtype=np.float32)
                    track_features = track_df.iloc[i][self.track_cols].values.astype(np.float32)

                    if np.isnan(point_features).any() or np.isnan(track_features).any():
                        print(f"警告: 在航迹 {track_id} 的第 {i} 行发现NaN值，已跳过该时间步。")
                        continue
                    
                    tabular_features = np.concatenate([point_features, track_features, time_interval])

                    # 2. 构建预期的.npy文件名并尝试加载
                    expected_npy_file = file_pattern_base.format(track_id=track_id, label=label, point_index=i+1)
                    rd_map_path = os.path.join(current_track_path, expected_npy_file)

                    try:
                        rd_map_raw = np.load(rd_map_path)
                    except FileNotFoundError:
                        # 如果文件未找到，创建零矩阵作为占位符
                        if expected_npy_file not in err_list:
                            err_list.append(expected_npy_file)
                            print(f"警告: RD图文件未找到: {expected_npy_file}。将使用零矩阵代替。")
                        # 尺寸可以是一个小值，因为后续的transform会将其resize
                        rd_map_raw = np.zeros((32, 32), dtype=np.uint8)

                    # 3. 将该时间步的数据（无论RD图是否存在）添加到序列中
                    track_steps.append({
                        'rd_map_path': rd_map_path,
                        'rd_map_raw': rd_map_raw,
                        'tabular_features': tabular_features,
                    })

                except IndexError:
                    print(f"警告: 在航迹 {track_id} 的第 {i} 行发生索引错误，提前结束该航迹处理。")
                    break # 如果发生索引错误，则停止处理此航迹的后续步骤
                except Exception as e:
                    print(f"提取特征时发生未知错误，在航迹 {track_id} 的第 {i} 行: {e}。已跳过该时间步。")
                    continue

            # 只保留有足够时间步的航迹（与之前相同）
            if len(track_steps) >= 2:
                self.tracks.append({
                    'track_id': track_id,
                    'label': label,
                    'steps': track_steps,
                    'length': len(track_steps)
                })

    def _create_label_map(self):
        """创建标签映射"""
        unique_labels = sorted(list(set(track['label'] for track in self.tracks)))
        self.label_map = {label: i for i, label in enumerate(unique_labels)}

    def _create_tabular_scaler(self,scaler_path):
        """计算表格数据的均值和标准差用于标准化"""
        all_features = []
        for track in self.tracks:
            for step in track['steps']:
                all_features.append(step['tabular_features'])

        all_features = np.array(all_features)
        self.tabular_scaler = {
            'mean': np.mean(all_features, axis=0),
            'std': np.std(all_features, axis=0)
        }
        # 防止标准差为0
        self.tabular_scaler['std'][self.tabular_scaler['std'] == 0] = 1.0
        np.savez(scaler_path, 
             mean=self.tabular_scaler['mean'], 
             std=self.tabular_scaler['std'])
        print(f"表格数据标准化参数已保存到: {scaler_path}")

    def _load_tabular_scaler(self,scaler_path):
        try:
            scaler_data = np.load(scaler_path)
            self.tabular_scaler = {'mean': scaler_data['mean'], 'std': scaler_data['std']}
            print("成功加载表格数据标准化参数。")
        except FileNotFoundError:
            print(f"错误: 找不到标准化参数文件 {scaler_path}。请先运行修改后的train.py生成该文件。")
            return

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        track_length = track['length']
        label = self.label_map[track['label']]

        # 初始化序列容器
        rd_map_sequence = []
        tabular_sequence = []

        # 处理每个时间步
        for step in track['steps']:
            # 处理RD图
            rd_map_raw = step['rd_map_raw']
            rd_map_normalized = (rd_map_raw - np.min(rd_map_raw)) / (np.max(rd_map_raw) - np.min(rd_map_raw) + 1e-8)
            rd_map_uint8 = (rd_map_normalized * 255).astype(np.uint8)
            image = Image.fromarray(rd_map_uint8, 'L')

            if self.transform:
                image = self.transform(image)

            rd_map_sequence.append(image)

            # 处理表格特征
            tabular_features = step['tabular_features']
            if self.tabular_scaler:
                tabular_features = (tabular_features - self.tabular_scaler['mean']) / self.tabular_scaler['std']

            tabular_sequence.append(torch.tensor(tabular_features, dtype=torch.float32))

        # 填充到最大序列长度
        pad_length = self.max_seq_len - len(rd_map_sequence)
        if pad_length > 0:
            # 填充RD图
            for _ in range(pad_length):
                rd_map_sequence.append(torch.zeros_like(rd_map_sequence[0]))
                tabular_sequence.append(torch.zeros_like(tabular_sequence[0]))

        # 转换为张量
        rd_map_sequence = torch.stack(rd_map_sequence, dim=0)  # (T, C, H, W)
        tabular_sequence = torch.stack(tabular_sequence, dim=0)  # (T, F)

        return {
            'rd_map_sequence': rd_map_sequence,
            'tabular_sequence': tabular_sequence,
            'length': track_length,  # 实际长度(未填充前)
            'label': label
        }


def get_vit_transforms(img_size=(224, 224)):
    """为Vision Transformer模型定义图像转换"""
    data_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return data_transforms