
import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
sys.path.append(os.path.join(BASE_DIR,'../src')) 

from train import main as run_training

def main():
    parser = argparse.ArgumentParser(description="训练部分")

    parser.add_argument('--data_root_dir', type=str, default=os.path.join(BASE_DIR,'../'), help='数据根目录')
    parser.add_argument('--model_save_dir', type=str, default=os.path.join(BASE_DIR,'../src/model_save'))
    parser.add_argument('--log_dir', type=str, default=os.path.join(BASE_DIR,'../src/logs'))
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_seq_len', type=int, default=30)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    args = parser.parse_args()

    config = {
        'data_root_dir': args.data_root_dir,
        'img_size': (224, 224),  
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

    run_training(config)

if __name__ == '__main__':
    main()