import os
import pandas as pd
from flask import Flask, jsonify, request,redirect
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import re


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


APP_CONFIG = {
    'TRAIN_ROOT_DIR': BASE_DIR,
    'CUSTOMER_ROOT_DIR': os.path.join(BASE_DIR, '测试数据') 
}

app = Flask(__name__)

CORS(app) 
available_batches = {}


def scan_and_cache_batches():

    global available_batches
    available_batches = {} 

    sources = {
        'train': APP_CONFIG['TRAIN_ROOT_DIR'],
        'customer': APP_CONFIG['CUSTOMER_ROOT_DIR']
    }
    
    for source_name, root_dir in sources.items():
        available_batches[source_name] = []
        track_dir = os.path.join(root_dir, '航迹')
        
        if not os.path.isdir(track_dir):
            print(f"警告: 在数据源 '{source_name}' 中未找到航迹目录: {track_dir}")
            continue
    
        found_ids = set()

        pattern1 = re.compile(r'Tracks_(\d+)_(\d+)_(\d+)\.txt')
        pattern2 = re.compile(r'Tracks_(\d+)_(\d+)\.txt')

        for filename in os.listdir(track_dir):
            match1 = pattern1.match(filename)
            match2 = pattern2.match(filename)

            if source_name == 'train' and match1:
                batch_id, label, _ = match1.groups()
                if batch_id not in found_ids:
                    available_batches[source_name].append({'id': batch_id, 'label': label})
                    found_ids.add(batch_id)
            elif source_name == 'customer' and match2:
                batch_id, _ = match2.groups()
                if batch_id not in found_ids:
                    available_batches[source_name].append({'id': batch_id})
                    found_ids.add(batch_id)

        available_batches[source_name].sort(key=lambda x: int(x['id']))
    
    total_found = len(available_batches.get('train', [])) + len(available_batches.get('customer', []))
    print(f"扫描完成，共找到 {total_found} 个可用批号。")


def find_file_by_batch(directory, prefix, batch_id):
    """
    [改动] 使用更灵活的正则表达式来查找文件，无论中间有多少部分。
    """
    if not os.path.isdir(directory):
        return None

    pattern = re.compile(rf'{prefix}_{batch_id}(?:_\d+)*\.txt')
    for filename in os.listdir(directory):
        if pattern.match(filename):
            return os.path.join(directory, filename)
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/paths', methods=['GET', 'POST'])
def handle_paths():
    if request.method == 'GET':
        return jsonify(APP_CONFIG)
    if request.method == 'POST':
        data = request.get_json()
        if data.get('TRAIN_ROOT_DIR'): APP_CONFIG['TRAIN_ROOT_DIR'] = data.get('TRAIN_ROOT_DIR')
        if data.get('CUSTOMER_ROOT_DIR'): APP_CONFIG['CUSTOMER_ROOT_DIR'] = data.get('CUSTOMER_ROOT_DIR')
        scan_and_cache_batches()
        return jsonify({'status': 'success', 'message': '路径已成功更新并重新扫描。', 'paths': APP_CONFIG})

@app.route('/scan_dataset', methods=['GET'])
def scan_dataset_endpoint():
    print("收到扫描数据集请求...")
    errors = []
    required_dirs = {
        "训练-点迹": os.path.join(APP_CONFIG['TRAIN_ROOT_DIR'], '点迹'),
        "训练-航迹": os.path.join(APP_CONFIG['TRAIN_ROOT_DIR'], '航迹'),
        "自定义-点迹": os.path.join(APP_CONFIG['CUSTOMER_ROOT_DIR'], '点迹'),
        "自定义-航迹": os.path.join(APP_CONFIG['CUSTOMER_ROOT_DIR'], '航迹'),
    }
    for name, path in required_dirs.items():
        if not os.path.isdir(path):
            errors.append(f"提示: 目录 '{path}' ({name}) 未找到。")
    scan_and_cache_batches()
    total_batches = len(available_batches.get('train', [])) + len(available_batches.get('customer', []))
    if not total_batches and not errors:
         return jsonify({'status': 'warning', 'message': '核心目录结构似乎存在，但在航迹目录中未找到任何有效批号文件。'}), 200
    success_msg = f'扫描到 {total_batches} 个有效批号 (训练: {len(available_batches.get("train", []))}, 自定义: {len(available_batches.get("customer", []))})。'
    final_message = success_msg + (" " + " | ".join(errors) if errors else "")
    return jsonify({'status': 'success', 'message': final_message})

@app.route('/batches', methods=['GET'])
def get_batches_endpoint():
    if not available_batches:
        scan_and_cache_batches()
    return jsonify({'sources': available_batches})

def get_data_for_batch(data_type, request_data):
    batch_id = request_data.get('batch')
    source = request_data.get('source')
    if not batch_id or not source: return jsonify({'error': '未提供批号或数据源'}), 400
    root_dir = APP_CONFIG.get(f'{source.upper()}_ROOT_DIR')
    if not root_dir: return jsonify({'error': f'无效的数据源: {source}'}), 400
    if data_type == 'track':
        data_dir = os.path.join(root_dir, '航迹')
        prefix = 'Tracks'
        error_msg = f'找不到批号 {batch_id} (源: {source}) 对应的航迹文件'
    elif data_type == 'point':
        data_dir = os.path.join(root_dir, '点迹')
        prefix = 'PointTracks'
        error_msg = f'找不到批号 {batch_id} (源: {source}) 对应的点迹文件'
    else: return jsonify({'error': '无效的数据类型'}), 500
    file_path = find_file_by_batch(data_dir, prefix, batch_id)
    if not file_path: return jsonify({'error': error_msg}), 404
    return file_path

@app.route('/track_data_table', methods=['POST'])
def get_track_data_table_endpoint():
    file_path_or_response = get_data_for_batch('track', request.get_json())
    if not isinstance(file_path_or_response, str): return file_path_or_response
    try:
        df = pd.read_csv(file_path_or_response, encoding='GBK')
        return jsonify({'data': df.to_dict('records')})
    except Exception as e: return jsonify({'error': f'读取或解析航迹文件时出错: {e}'}), 500

@app.route('/track_coordinates', methods=['POST'])
def get_track_coordinates_endpoint():
    """
    [改动] 根据源和文件内容，动态生成标签并返回。
    """
    request_data = request.get_json()
    source = request_data.get('source')
    batch_id = request_data.get('batch')

    file_path_or_response = get_data_for_batch('track', request_data)
    if not isinstance(file_path_or_response, str):
        return file_path_or_response

    try:
        df = pd.read_csv(file_path_or_response, encoding='GBK')
        
        main_label = ""
        coords_df = df[['滤波距离', '滤波方位']].rename(columns={'滤波距离': 'distance', '滤波方位': 'angle'})

        if source == 'train':
            batch_info = next((item for item in available_batches.get('train', []) if item['id'] == batch_id), None)
            if batch_info and 'label' in batch_info:
                main_label = f": label {batch_info['label']}"
        
        elif source == 'customer':
            if '识别结果' in df.columns:

                main_label = "(测试数据)"

                coords_df['label'] = df['识别结果'].astype(str)
            else:

                main_label = ": 待预测 (测试数据)"

        records = coords_df.to_dict('records')
        return jsonify({'coordinates': records, 'main_label': main_label})

    except Exception as e:
        return jsonify({'error': f'处理航迹坐标时出错: {e}'}), 500

@app.route('/point_coordinates', methods=['POST'])
def get_point_coordinates_endpoint():
    file_path_or_response = get_data_for_batch('point', request.get_json())
    if not isinstance(file_path_or_response, str): return file_path_or_response
    try:
        df = pd.read_csv(file_path_or_response, encoding='GBK')
        df_coords = df[['距离', '俯仰']].rename(columns={'距离': 'x', '俯仰': 'y'})
        return jsonify({'scatter_data': df_coords.to_dict('records')})
    except Exception as e: return jsonify({'error': f'处理点迹坐标时出错: {e}'}), 500

@app.route('/training')
def train_redirect():
    return redirect("http://127.0.0.1:7870", code=302)

if __name__ == '__main__':
    scan_and_cache_batches()
    app.run(host='0.0.0.0', port=5002, debug=True)