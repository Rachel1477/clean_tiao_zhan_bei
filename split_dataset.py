import os
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

def split_dataset(base_dir, output_dir, train_ratio=0.8):
    """
    将结构化的NPY数据集划分为训练集和验证集。
    这个函数假设原始数据集的结构如下:
    base_dir/
    ├── 1_Label_1/
    │   ├── 1_Label_1_Point_1.npy
    │   └── ...
    ├── 2_Label_3/
    │   └── ...
    └── ...
    它会创建一个新的目录结构:
    output_dir/
    ├── train/
    │   ├── 1_Label_1/
    │   └── ...
    └── val/
    │   ├── ...
    │   └── ...
    其中，base_dir和output_dir都在你的项目根目录上，请确保你的目录组织如上 ！！！！！
    """

    print(f"（调试输出）开始处理，源目录: {base_dir}")
    print(f"（调试输出）输出目录: {output_dir}")
    print(f"（调试输出）训练集比例: {train_ratio}, 验证集比例: {(1 - train_ratio)/2},纯测试集比例{(1 - train_ratio)/2}")


    if not os.path.isdir(base_dir):
        print(f"错误: 找不到 '{base_dir}' ")
        return


    train_path = os.path.join(output_dir, 'train')
    val_path = os.path.join(output_dir, 'val')
    test_path = os.path.join(output_dir, 'test')

    if os.path.exists(output_dir):
        print(f"（调试输出）  '{output_dir}' 已存在，其中的内容可能会被覆盖。")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    print("（调试输出）成功创建 'train' 'val' 'test' 目录。")


    tracks_by_class = defaultdict(list)
    

    main_target_labels = {'1', '2', '3', '4'}

    print("（调试输出）开始扫描并按类别收集航迹文件夹...")
    for track_folder_name in os.listdir(base_dir):
        # 下面的代码用于解析文件夹名以获取标签
        parts = track_folder_name.split('_')
        if len(parts) >= 3 and parts[1] == 'Label':
            class_label = parts[2]
            
            # 如果只想训练比赛要求的4类，取消下面这行注释，这将会忽略类别5-杂波，类别6-未定义数据
            if class_label not in main_target_labels:
                continue


            tracks_by_class[class_label].append(track_folder_name)
        else:
            print(f"跳过不符合命名规范的文件夹: {track_folder_name}")

    print("\n（调试输出）类别统计如下:")
    for label, tracks in tracks_by_class.items():
        print(f"  - 类别 '{label}': 找到 {len(tracks)} 个航迹样本")

    print("\n（调试输出）开始划分数据集并移动文件夹...")
    
    total_moved_train = 0
    total_moved_val = 0
    total_moved_test=0

    for class_label, track_list in tqdm(tracks_by_class.items(), desc="处理类别"):
        # 对每个类别的航迹列表进行随机打乱
        random.shuffle(track_list)
    
        test_index =int((len(track_list)-len(track_list) * train_ratio)/2)
        val_index =int((len(track_list)-len(track_list) * train_ratio))
        # 划分训练集和验证集
        train_tracks = track_list[val_index:]
        val_tracks = track_list[test_index:val_index]
        test_tracks=track_list[:test_index]
        # 移动训练集文件夹
        for track_folder in train_tracks:
            src_path = os.path.join(base_dir, track_folder)
            dest_path = os.path.join(train_path, track_folder)
            shutil.move(src_path, dest_path)
            total_moved_train += 1
            
        # 移动验证集文件夹
        for track_folder in val_tracks:
            src_path = os.path.join(base_dir, track_folder)
            dest_path = os.path.join(val_path, track_folder)
            shutil.move(src_path, dest_path)
            total_moved_val += 1
        
        for track_folder in test_tracks:
            src_path = os.path.join(base_dir, track_folder)
            dest_path = os.path.join(test_path, track_folder)
            shutil.move(src_path, dest_path)
            total_moved_test += 1
    
    print("\n（调试输出）数据集划分完成！")
    print(f"（调试输出）请检查'{output_dir}' 目录是否正常，如果正常，就开始你的后续操作。")


if __name__ == '__main__':
    # 源目录，即包含所有航迹子文件夹的目录
    SOURCE_DIRECTORY = 'NPY_Output_Structured'
    # 目标目录，用于存放划分好的 train 和 val 文件夹
    OUTPUT_DIRECTORY = 'dataset'
    # 训练集比例
    TRAIN_RATIO = 0.7
    split_dataset(SOURCE_DIRECTORY, OUTPUT_DIRECTORY, TRAIN_RATIO)