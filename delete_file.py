import os
import sys

# --- 配置区 ---

# 1. 设置你要扫描的根目录
#    Windows示例: 'C:/Users/YourUser/Desktop/data'
#    Linux/macOS示例: '/home/user/data'
root_directory = '/home/rachel/Desktop/finual_solution' 

# 2. 安全开关：演练模式 (Dry Run)
#    True  -> 只打印将要删除的文件，不执行任何删除操作 (强烈建议首次运行时使用!)
#    False -> 真正执行删除操作
dry_run = False

# --- 脚本主逻辑 (一般无需修改) ---

def delete_specific_label_files(target_dir, is_dry_run):
    """
    遍历目录，删除标签为5或6的PointTracks文件。
    """
    # 检查根目录是否存在
    if not os.path.isdir(target_dir):
        print(f"错误：目录 '{target_dir}' 不存在或不是一个有效的目录。")
        print("请在脚本中正确设置 'root_directory' 变量。")
        sys.exit(1) # 退出脚本

    files_to_delete = []
    
    print(f"正在扫描目录: {target_dir}")
    # os.walk 会遍历指定目录下的所有子目录和文件
    for dirpath, _, filenames in os.walk(target_dir):
        for filename in filenames:
            # 筛选出可能符合条件的文件，提高效率
            if filename.startswith("Tracks_") and filename.endswith(".txt"):
                try:
                    # 去掉 .txt 后缀，然后按 '_' 分割
                    parts = os.path.splitext(filename)[0].split('_')
                    # PointTracks_{track_id}_{label}_{track_len} -> 应该有4个部分
                    if len(parts) == 4:
                        label = parts[2]
                        # 检查label是否为 '5' 或 '6'
                        if label in ('5', '6'):
                            # 构建文件的完整路径
                            full_path = os.path.join(dirpath, filename)
                            files_to_delete.append(full_path)
                except Exception as e:
                    # 如果文件名格式不规范，打印错误并跳过
                    print(f"跳过格式不正确的文件: {filename} (错误: {e})")
    
    # --- 执行删除或打印 ---
    
    if not files_to_delete:
        print("\n扫描完成，没有找到标签为 5 或 6 的文件。")
        return

    print("-" * 50)
    if is_dry_run:
        print(f"[演练模式] 找到了 {len(files_to_delete)} 个将要被删除的文件：")
        for f_path in files_to_delete:
            print(f"  - {f_path}")
        print("\n[演练模式] 未执行任何删除操作。")
        print("要真正删除文件，请将脚本中的 'dry_run' 设置为 False。")
    else:
        print(f"即将删除 {len(files_to_delete)} 个文件...")
        deleted_count = 0
        error_count = 0
        for f_path in files_to_delete:
            try:
                os.remove(f_path)
                print(f"已删除: {f_path}")
                deleted_count += 1
            except OSError as e:
                print(f"删除失败: {f_path} (错误: {e})")
                error_count += 1
        print("-" * 50)
        print("操作完成。")
        print(f"成功删除: {deleted_count} 个文件")
        if error_count > 0:
            print(f"删除失败: {error_count} 个文件")


# --- 运行脚本 ---
if __name__ == "__main__":
    delete_specific_label_files(root_directory, dry_run)