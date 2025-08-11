#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_rename.py

功能：
1) 在目录中（可递归）把名字里包含 "Label_<数字>" 的那段删除（包含文件和文件夹）。
   例如:
     1_Label_1_Point_1.npy -> 1_Point_1.npy
     1_Label_1 -> 1_

   处理规则（与示例一致）：
     - 如果 "Label_n" 两边都有 '_'，将整段（包括两边的 '_'）替换成单个 '_'：
         "a_Label_1_b" -> "a_b"
     - 如果 "Label_n" 只有左侧有 '_'（即位于结尾），删除 "Label_n"，保留左侧下划线：
         "a_Label_1" -> "a_"
     - 如果 "Label_n" 只有右侧有 '_'（即位于起始），删除 "Label_n"，保留右侧下划线：
         "Label_1_a" -> "_a"
     - 如果两侧都没有下划线，直接删除 "Label_n"。
2) 对文件名形如 Tracks_x_y_z.* 的，删除第二个数字 y（以及前面的下划线），变为 Tracks_x_z.*。
   例如:
     Tracks_2_1_15.txt -> Tracks_2_15.txt

安全：
- 默认是 dry-run（只打印将要改动的内容）。
- 使用 --commit 才会真正改名。
- 如果目标名已存在，会自动在文件名后加数字后缀避免覆盖（例如 name.txt -> name_1.txt）。

用法示例：
  # 仅演练（默认）
  python3 batch_rename.py /path/to/dir

  # 真正执行（递归）
  python3 batch_rename.py /path/to/dir --commit

  # 只在指定目录，不递归子目录
  python3 batch_rename.py /path/to/dir --no-recursive --commit
"""

import re
import argparse
from pathlib import Path

def remove_label_parts(raw_name: str) -> str:
    """
    删除名字中的 Label_<digits> 部分，按说明处理相邻下划线。
    raw_name: 文件名（不含扩展名）或目录名
    返回处理后的名字（不含扩展名）。
    """
    pattern = re.compile(r'Label_(\d+)')
    s = raw_name
    # 迭代所有匹配，从左到右处理（因为替换可能改变索引）
    i = 0
    while True:
        m = pattern.search(s, i)
        if not m:
            break
        start, end = m.start(), m.end()
        left = s[start-1] if start-1 >= 0 else ''
        right = s[end] if end < len(s) else ''

        if left == '_' and right == '_':
            # 替换 left + match + right 为 单个 _
            s = s[:start-1] + '_' + s[end+1:]
            # 新位置 i = start (继续搜索后面的)
            i = start
        elif left == '_' and right != '_':
            # 保留左侧下划线，删除 match 本身
            s = s[:start] + s[end:]
            i = start  # 继续
        elif left != '_' and right == '_':
            # 保留右侧下划线，删除 match 本身
            s = s[:start] + s[end:]
            i = start
        else:
            # 两侧都没有下划线，直接删除 match
            s = s[:start] + s[end:]
            i = start
    return s

def rename_tracks_second_number(name_no_ext: str) -> str:
    """
    对 Tracks_ 开头并且下划线分段 >=4 的名字，删除第二个数字及其前导下划线。
    例如 Tracks_2_1_15 -> Tracks_2_15
    如果不匹配格式，返回原名。
    """
    parts = name_no_ext.split('_')
    if len(parts) >= 4 and parts[0] == 'PointTracks':
        # 保留 parts[0], parts[1], 然后拼接 parts[3...]
        new_parts = [parts[0], parts[1]] + parts[3:]
        return '_'.join(new_parts)
    return name_no_ext

def unique_path(target: Path) -> Path:
    """
    如果 target 已存在，自动在名字后加 _1, _2, ... 直到不冲突。
    保持扩展名不变。
    """
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1

def safe_rename(src: Path, dst: Path, do_commit: bool):
    """
    安全地重命名文件或文件夹，避免覆盖（会自动改名以避免覆盖）。
    """
    if src.resolve() == dst.resolve():
        return  # 相同路径，无需操作
    dst_final = unique_path(dst)
    if do_commit:
        src.rename(dst_final)
        print(f"[RENAMED] {src} -> {dst_final}")
    else:
        print(f"[DRY-RUN] {src} -> {dst_final}")

def process_directory(root: Path, recursive=True, do_commit=False):
    """
    遍历 root（可递归），对文件和目录应用规则 1（删除 Label_n）和规则 2（Tracks 文件）。
    目录采用 bottom-up（先内层目录再外层）以避免路径变更导致的问题。
    """
    if recursive:
        walker = root.rglob('*')
        # But we need directories bottom-up. Use os.walk-like approach with topdown=False via Path.rglob is not ordered.
        # Simpler: collect all paths, sort directories by depth descending.
        all_paths = [p for p in root.glob('**/*')]
        # handle directories first (deepest first), then files
        dirs = [p for p in all_paths if p.is_dir()]
        files = [p for p in all_paths if p.is_file()]
        dirs.sort(key=lambda p: len(p.parts), reverse=True)  # deepest first
    else:
        dirs = [p for p in root.iterdir() if p.is_dir()]
        files = [p for p in root.iterdir() if p.is_file()]

    # 先处理文件（文件名可能受两条规则影响）
    for f in files:
        orig_name = f.name
        stem = f.stem
        suffix = f.suffix  # 包含点，例如 .txt, .npy

        # 规则1：删除 Label_n
        new_stem = remove_label_parts(stem)

        # 规则2：Tracks 特殊处理（基于删除 Label 后或原名都可）
        new_stem = rename_tracks_second_number(new_stem)

        if new_stem != stem:
            new_name = new_stem + suffix
            dst = f.with_name(new_name)
            safe_rename(f, dst, do_commit)

    # 然后处理目录（从内到外）
    for d in dirs:
        orig_name = d.name
        new_name = remove_label_parts(orig_name)
        # 注意：Tracks 规则通常用于文件名，这里不对目录应用 Tracks 规则
        if new_name != orig_name:
            dst = d.with_name(new_name)
            safe_rename(d, dst, do_commit)

def main():
    parser = argparse.ArgumentParser(description='批量重命名：删除 Label_<n>，以及 Tracks_*_*_*. 文件。')
    parser.add_argument('root', type=str, help='目标目录路径')
    parser.add_argument('--no-recursive', action='store_true', help='只在根目录操作，不递归子目录')
    parser.add_argument('--commit', action='store_true', help='实际执行重命名。默认仅 dry-run（打印）。')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print("错误：root 必须是存在的目录路径。")
        return

    print(f"{'执行' if args.commit else '演练（dry-run）'}：root = {root}, recursive = {not args.no_recursive}")
    process_directory(root, recursive=not args.no_recursive, do_commit=args.commit)
    print("完成。")

if __name__ == '__main__':
    main()
