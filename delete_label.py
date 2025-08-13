
import re
import argparse
from pathlib import Path

def remove_label_parts(raw_name: str) -> str:
    pattern = re.compile(r'_Label_(\d+)')
    s = raw_name

    i = 0
    while True:
        m = pattern.search(s, i)
        if not m:
            break
        start, end = m.start(), m.end()
        left = s[start-1] if start-1 >= 0 else ''
        right = s[end] if end < len(s) else ''
        if left == '_' and right == '_':
            s = s[:start-1] + '_' + s[end+1:]
            i = start
        elif left == '_' and right != '_':
            s = s[:start] + s[end:]
            i = start  
        elif left != '_' and right == '_':
            s = s[:start] + s[end:]
            i = start
        else:
            s = s[:start] + s[end:]
            i = start
    return s

def rename_tracks_second_number(name_no_ext: str) -> str:
    parts = name_no_ext.split('_')
    if len(parts) >= 4 and parts[0] == 'Tracks':
        new_parts = [parts[0], parts[1]] + parts[3:]
        return '_'.join(new_parts)
    return name_no_ext

def unique_path(target: Path) -> Path:

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

    if src.resolve() == dst.resolve():
        return  # 相同路径，无需操作
    dst_final = unique_path(dst)
    if do_commit:
        src.rename(dst_final)
        print(f"[RENAMED] {src} -> {dst_final}")
    else:
        print(f"[DRY-RUN] {src} -> {dst_final}")

def process_directory(root: Path, recursive=True, do_commit=False):

    if recursive:
        walker = root.rglob('*')
        all_paths = [p for p in root.glob('**/*')]

        dirs = [p for p in all_paths if p.is_dir()]
        files = [p for p in all_paths if p.is_file()]
        dirs.sort(key=lambda p: len(p.parts), reverse=True)  
    else:
        dirs = [p for p in root.iterdir() if p.is_dir()]
        files = [p for p in root.iterdir() if p.is_file()]

    for f in files:
        orig_name = f.name
        stem = f.stem
        suffix = f.suffix  
        new_stem = remove_label_parts(stem)
        new_stem = rename_tracks_second_number(new_stem)

        if new_stem != stem:
            new_name = new_stem + suffix
            dst = f.with_name(new_name)
            safe_rename(f, dst, do_commit)
    for d in dirs:
        orig_name = d.name
        new_name = remove_label_parts(orig_name)
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
