import os
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

def split_dataset(base_dir, output_dir, train_ratio=0.8):
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

        parts = track_folder_name.split('_')
        if len(parts) >= 3 and parts[1] == 'Label':
            class_label = parts[2]
        
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
        random.shuffle(track_list)
    
        test_index =int((len(track_list)-len(track_list) * train_ratio)/2)
        val_index =int((len(track_list)-len(track_list) * train_ratio))

        train_tracks = track_list[val_index:]
        val_tracks = track_list[test_index:val_index]
        test_tracks=track_list[:test_index]

        for track_folder in train_tracks:
            src_path = os.path.join(base_dir, track_folder)
            dest_path = os.path.join(train_path, track_folder)
            shutil.move(src_path, dest_path)
            total_moved_train += 1

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
    SOURCE_DIRECTORY = 'NPY_Output_Structured'
    OUTPUT_DIRECTORY = 'DRmap'
    # 训练集比例，你可以自己定义
    TRAIN_RATIO = 0.7
    split_dataset(SOURCE_DIRECTORY, OUTPUT_DIRECTORY, TRAIN_RATIO)