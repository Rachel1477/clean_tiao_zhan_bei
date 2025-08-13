# --- START OF FILE run_training_process.py ---
import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #当前文件的路径
sys.path.append(os.path.join(BASE_DIR,'../src')) 

from train import main as run_training

def main():
    parser = argparse.ArgumentParser(description="Run training process with specified configurations.")
    
    # 定义所有可配置的参数
    parser.add_argument('--data_root_dir', type=str, default=os.path.join(BASE_DIR,'../'), help='Root directory of the dataset.')
    parser.add_argument('--model_save_dir', type=str, default=os.path.join(BASE_DIR,'../src/model_save'), help='Directory to save models.')
    parser.add_argument('--log_dir', type=str, default=os.path.join(BASE_DIR,'../src/logs'), help='Directory for TensorBoard logs.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--max_seq_len', type=int, default=30, help='Maximum sequence length.')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of warmup epochs.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value.')

    args = parser.parse_args()

    # 将命令行参数组织成 train.py 中 main 函数期望的字典格式
    config = {
        'data_root_dir': args.data_root_dir,
        'img_size': (224, 224),  # 保持固定或也可以设为参数
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'patience': args.patience,
        'model_save_dir': args.model_save_dir,
        'log_dir': args.log_dir,
        'grad_clip': args.grad_clip,
        'warmup_epochs': args.warmup_epochs,
        'max_seq_len': args.max_seq_len,
        'mod':'train'
    }

    # 调用导入的训练函数
    run_training(config)

if __name__ == '__main__':
    main()