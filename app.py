import os
import pandas as pd
from flask import Flask, jsonify, request
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import re

# --- 配置 ---
# 设置包含“点迹”、“航迹”等文件夹的数据集根目录
# 假设app.py与这些文件夹在同一级目录
DATA_ROOT_DIR = './' 
POINT_DIR = os.path.join(DATA_ROOT_DIR, '点迹')
TRACK_DIR = os.path.join(DATA_ROOT_DIR, '航迹')
RAW_DIR = os.path.join(DATA_ROOT_DIR, '原始回波')
DATASET_DIR = os.path.join(DATA_ROOT_DIR, 'dataset/train')  #这个DR map没有被用到可视化，但是我们认为没有DR map的是不完整的数据


# Flask 应用初始化
app = Flask(__name__)
# 允许所有来源的跨域请求，方便前端开发
CORS(app) 

# 下面这个全局变量，用于缓存扫描结果
available_batches = []


def scan_and_cache_batches():
    """
    扫描航迹目录，提取所有可用的航迹批号并缓存。
    文件名格式: Tracks_航迹批号_目标类型标签_航迹长度.txt
    """
    global available_batches
    available_batches = []
    if not os.path.isdir(TRACK_DIR):
        return
        
    # 使用正则表达式从文件名中提取航迹批号
    pattern = re.compile(r'Tracks_(\d+)_(\d+)_(\d+)\.txt')
    
    for filename in os.listdir(TRACK_DIR):
        match = pattern.match(filename)
        if match:
            batch_id = match.group(1)
            if batch_id not in available_batches:
                available_batches.append(batch_id)
    
    # 按数字大小排序
    available_batches.sort(key=int)
    print(f"扫描完成，找到 {len(available_batches)} 个可用批号。")


def find_file_by_batch(directory, prefix, batch_id):
    """
    根据批号在指定目录中查找对应的文件。
    参数:
    directory (str): 要搜索的目录 (e.g., '点迹', '航迹')
    prefix (str): 文件名前缀 (e.g., 'PointTracks', 'Tracks')
    batch_id (str): 航迹批号
    返回:
    str or None: 找到的文件的完整路径，如果找不到则返回None。
    """
    if not os.path.isdir(directory):
        return None
    
    # 构造一个更灵活的正则表达式，只要求批号匹配
    pattern = re.compile(rf'{prefix}_{batch_id}_\d+_\d+\.txt')

    for filename in os.listdir(directory):
        if pattern.match(filename):
            return os.path.join(directory, filename)
    return None


#  API 端点 (Endpoints)
@app.route('/')
def index():
    """
    当用户访问根URL (本地是http://127.0.0.1:5000) 时，
    渲染并返回 templates/index.html 文件。
    """
    return render_template('index.html')

@app.route('/scan_dataset', methods=['GET'])
def scan_dataset_endpoint():
    """
    API: 扫描数据集的完整性。
    """
    print("收到扫描数据集请求...")
    errors = []
    
    # 检查核心目录是否存在
    required_dirs = {
        "点迹数据目录": POINT_DIR,
        "航迹数据目录": TRACK_DIR,
        "划分后数据集目录": DATASET_DIR
    }
    
    for name, path in required_dirs.items():
        if not os.path.isdir(path):
            errors.append(f"错误: {name} '{path}' 未找到！")
            
    if errors:
        return jsonify({'status': 'error', 'message': " | ".join(errors)}), 404

    # 扫描并更新批号列表缓存
    scan_and_cache_batches()

    if not available_batches:
         return jsonify({'status': 'warning', 'message': '核心目录存在，但在航迹目录中未找到任何有效批号文件。'}), 200

    return jsonify({'status': 'success', 'message': f'数据集完整，扫描到 {len(available_batches)} 个有效批号。'})


@app.route('/batches', methods=['GET'])
def get_batches_endpoint():
    """
    API: 获取所有可用的航迹批号列表。
    """
    # 如果缓存为空，则先扫描一次
    if not available_batches:
        scan_and_cache_batches()
        
    return jsonify({'batches': available_batches})


@app.route('/track_data_table', methods=['POST'])
def get_track_data_table_endpoint():
    """
    API: 根据批号获取航迹文件的表格数据。
    """
    data = request.get_json()
    batch_id = data.get('batch')
    if not batch_id:
        return jsonify({'error': '未提供批号'}), 400

    track_file_path = find_file_by_batch(TRACK_DIR, 'Tracks', batch_id)
    if not track_file_path:
        return jsonify({'error': f'找不到批号 {batch_id} 对应的航迹文件'}), 404

    try:
        df = pd.read_csv(track_file_path, encoding='GBK')
        # 将DataFrame转换为字典列表，方便前端处理
        records = df.to_dict('records')
        return jsonify({'data': records})
    except Exception as e:
        return jsonify({'error': f'读取或解析航迹文件时出错: {e}'}), 500


@app.route('/track_coordinates', methods=['POST'])
def get_track_coordinates_endpoint():
    """
    API: 根据批号获取用于雷达图显示的航迹坐标。
    """
    data = request.get_json()
    batch_id = data.get('batch')
    if not batch_id:
        return jsonify({'error': '未提供批号'}), 400

    track_file_path = find_file_by_batch(TRACK_DIR, 'Tracks', batch_id)
    if not track_file_path:
        return jsonify({'error': f'找不到批号 {batch_id} 对应的航迹文件'}), 404

    try:
        df = pd.read_csv(track_file_path, encoding='GBK')
        # 选择需要的列并重命名
        df_coords = df[['滤波距离', '滤波方位']].rename(columns={'滤波距离': 'distance', '滤波方位': 'angle'})
        
        # 将距离从米转换为公里（如果需要，取决于前端雷达图的刻度）
        # df_coords['distance'] = df_coords['distance'] / 1000

        records = df_coords.to_dict('records')
        return jsonify({'coordinates': records})
    except Exception as e:
        return jsonify({'error': f'处理航迹坐标时出错: {e}'}), 500


@app.route('/point_coordinates', methods=['POST'])
def get_point_coordinates_endpoint():
    """
    API: 根据批号获取用于散点图显示的点迹坐标。
    """
    data = request.get_json()
    batch_id = data.get('batch')
    if not batch_id:
        return jsonify({'error': '未提供批号'}), 400

    point_file_path = find_file_by_batch(POINT_DIR, 'PointTracks', batch_id)
    if not point_file_path:
        return jsonify({'error': f'找不到批号 {batch_id} 对应的点迹文件'}), 404

    try:
        df = pd.read_csv(point_file_path, encoding='GBK')
        # 选择需要的列并重命名
        # 散点图通常用x, y表示，这里我们用距离和俯仰角
        df_coords = df[['距离', '俯仰']].rename(columns={'距离': 'x', '俯仰': 'y'})
        records = df_coords.to_dict('records')
        return jsonify({'scatter_data': records})
    except Exception as e:
        return jsonify({'error': f'处理点迹坐标时出错: {e}'}), 500

# 运行应用
if __name__ == '__main__':
    # 在启动时扫描一次批号
    scan_and_cache_batches()
    # 启动Flask服务器
    # host='0.0.0.0' 允许局域网内其他设备访问
    # debug=True 会在代码修改后自动重启服务器，方便开发
    app.run(host='0.0.0.0', port=5000, debug=True)