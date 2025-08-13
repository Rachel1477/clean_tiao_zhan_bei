import argparse
import os
import sys

# --- Path Setup ---
# This ensures that the script can find the 'src' directory where apply.py is located.
# It assumes a folder structure like:
# project/
#  ├─ app/
#  │  ├─ run_prediction_process.py
#  │  └─ gradio_interface.py
#  └─ src/
#     └─ apply.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the 'src' directory to Python's path
sys.path.append(os.path.join(BASE_DIR, '../src'))

from apply import main as run_prediction

def main():
    parser = argparse.ArgumentParser(description="Run prediction process with specified configurations.")

    # --- CLI Arguments ---
    parser.add_argument('--test_data_root', type=str, required=True,default=os.path.join(BASE_DIR,'../测试数据'), help='Root directory of the test dataset.')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory where the trained model and artifacts are saved.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save prediction results.')
    
    # You can also expose model hyperparameters here if needed, but it's often better
    # to keep them fixed to match the training configuration loaded from apply.py.
    
    args = parser.parse_args()

    # --- Build CONFIG Dictionary for apply.py ---
    # The CONFIG in apply.py already contains the model's architectural parameters.
    # We just need to override the paths.
    config = {
        'test_data_root': args.test_data_root,
        'model_dir': args.model_dir,
        'output_dir': args.output_dir,
        'model_filename': 'best.pth',
        'img_size': (224, 224),
        
        # These architecture parameters must match the trained model.
        # They are defined in apply.py but we can list them here for clarity.
        'patch_size': 32,
        'embed_dim': 128,
        'depth': 3,
        'heads': 4,
        'mlp_dim': 256,
        'lstm_hidden_dim': 128,
        'lstm_num_layers': 2,
        'dropout': 0.1,
    }

    print("--- Starting Prediction Process ---")
    run_prediction(config)
    print("--- Prediction Process Finished ---")

if __name__ == '__main__':
    main()