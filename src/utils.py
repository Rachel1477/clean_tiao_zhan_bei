import pandas as pd
import numpy as np
from datetime import datetime

def parse_time_to_seconds(time_str):
    """
    将 'HH:MM:SS.micros' 格式的时间字符串解析为从当天零点开始的总秒数。

    参数:
    time_str (str): 时间字符串。

    返回:
    float: 总秒数。如果解析失败，返回 np.nan。
    """
    if not isinstance(time_str, str):
        return np.nan
    try:
        # datetime.strptime 不能直接解析超过6位的微秒，需要手动处理
        if '.' in time_str:
            parts = time_str.split('.')
            main_time_part = parts[0]
            micro_part = parts[1]
            # 截断微秒到6位
            micro_part = micro_part[:6]
            time_obj = datetime.strptime(f"{main_time_part}.{micro_part}", '%H:%M:%S.%f').time()
        else:
            time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
        
        return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    except (ValueError, IndexError):
        return np.nan

def clean_tabular_dataframe(df, numeric_cols):
    """
    对输入的Pandas DataFrame进行彻底的清洗。
    1. 将指定列强制转换为数值类型，无效值转为NaN。
    2. 用列均值填充NaN。
    3. 用0填充剩余的NaN（如果某列全是NaN）。

    参数:
    df (pd.DataFrame): 原始的Pandas DataFrame。
    numeric_cols (list): 需要清洗和转换为数值类型的列名列表。

    返回:
    pd.DataFrame: 清洗后的DataFrame。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是一个Pandas DataFrame。")

    # 1. 强制转换为数值类型
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"警告: 在DataFrame中找不到列 '{col}'。")
    
    # 2. 用列均值填充NaN
    # 我们只选择要处理的列进行操作
    subset_df = df[numeric_cols]
    if subset_df.isnull().values.any():
        # print(f"检测到NaN值，将用列均值填充。")
        # .copy() to avoid SettingWithCopyWarning
        df_cleaned = df.copy()
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(subset_df.mean(numeric_only=True))
    else:
        df_cleaned = df

    # 3. 双重保险：用0填充剩余的NaN
    df_cleaned.fillna(0, inplace=True)
    
    return df_cleaned


def calculate_time_features(df, time_col='time_seconds'):
    """
    在DataFrame中计算 delta_t 和 time_since_start 特征。
    假设DataFrame已经按时间排好序。

    参数:
    df (pd.DataFrame): 包含时间秒数列的DataFrame。
    time_col (str): 时间秒数列的列名。

    返回:
    pd.DataFrame: 增加了 'delta_t' 和 'time_since_start' 列的DataFrame。
    """
    if time_col not in df.columns:
        raise ValueError(f"在DataFrame中找不到时间列 '{time_col}'。")

    # 计算 delta_t (与上一行的时间差)
    # .diff() 会自动计算差值，第一行的结果是NaT/NaN
    df['delta_t'] = df[time_col].diff()
    # 将第一个点的delta_t填充为0
    df.loc[:, 'delta_t'] = df[time_col].diff().fillna(0)

    # 计算 time_since_start (与第一个点的时间差)
    first_time = df[time_col].iloc[0]
    df['time_since_start'] = df[time_col] - first_time
    
    return df


if __name__ == '__main__':
    # --- 这是一个用于测试该模块是否能正常工作的示例 ---
    print("开始测试 utils.py...")

    # 1. 测试时间解析
    print("\n--- 测试时间解析 ---")
    time_str1 = "12:33:49.652093"
    time_str2 = "17:20:22.670443123" # 超过6位微秒
    time_str3 = "18:00:00"
    time_str4 = "invalid_time"
    
    print(f"'{time_str1}' -> {parse_time_to_seconds(time_str1):.6f} seconds")
    print(f"'{time_str2}' -> {parse_time_to_seconds(time_str2):.6f} seconds (微秒被截断)")
    print(f"'{time_str3}' -> {parse_time_to_seconds(time_str3):.6f} seconds")
    print(f"'{time_str4}' -> {parse_time_to_seconds(time_str4)}")

    # 2. 测试数据清洗
    print("\n--- 测试数据清洗 ---")
    data = {
        'A': [1, 2, np.nan, 4, '5'],
        'B': [1.1, 2.2, 3.3, np.nan, 'invalid'],
        'C': [np.nan, np.nan, np.nan, np.nan, np.nan] # 整列都是NaN
    }
    df = pd.DataFrame(data)
    print("原始DataFrame:")
    print(df)
    
    cleaned_df = clean_tabular_dataframe(df, numeric_cols=['A', 'B', 'C'])
    print("\n清洗后的DataFrame:")
    print(cleaned_df)
    print("\n清洗后数据类型:")
    print(cleaned_df.dtypes)
    
    # 3. 测试时间特征计算
    print("\n--- 测试时间特征计算 ---")
    time_data = {'time_seconds': [100.0, 100.5, 100.8, 102.0]}
    time_df = pd.DataFrame(time_data)
    print("原始时间DataFrame:")
    print(time_df)
    
    time_featured_df = calculate_time_features(time_df)
    print("\n增加时间特征后的DataFrame:")
    print(time_featured_df)

    print("\nutils.py 模块看起来工作正常！")