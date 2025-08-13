import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../src'))

from apply import main as run_prediction

def main():
    parser = argparse.ArgumentParser(description="测试未知数据")

    parser.add_argument('--test_data_root', type=str, required=True,default=os.path.join(BASE_DIR,'../测试数据'), help='填入你的数据根目录')
    parser.add_argument('--model_dir', type=str, required=True, help='模型保存路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出路径')
    

    
    args = parser.parse_args()
    config = {
        'test_data_root': args.test_data_root,
        'model_dir': args.model_dir,
        'output_dir': args.output_dir,
        'model_filename': 'best.pth',
        'img_size': (224, 224),
        'patch_size': 32,
        'embed_dim': 128,
        'depth': 3,
        'heads': 4,
        'mlp_dim': 256,
        'lstm_hidden_dim': 128,
        'lstm_num_layers': 2,
        'dropout': 0.1,
    }
    run_prediction(config)

if __name__ == '__main__':
    main()