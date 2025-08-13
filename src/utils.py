import pandas as pd
import numpy as np
from datetime import datetime

def parse_time_to_seconds(time_str):
    if not isinstance(time_str, str):
        return np.nan
    try:
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
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是一个Pandas DataFrame。")

    # 1. 强制转换为数值类型
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"警告: 在DataFrame中找不到列 '{col}'。")
    
    subset_df = df[numeric_cols]
    if subset_df.isnull().values.any():
        df_cleaned = df.copy()
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(subset_df.mean(numeric_only=True))
    else:
        df_cleaned = df

    df_cleaned.fillna(0, inplace=True)
    
    return df_cleaned


def calculate_time_features(df, time_col='time_seconds'):
    if time_col not in df.columns:
        raise ValueError(f"在DataFrame中找不到时间列 '{time_col}'。")
    df['delta_t'] = df[time_col].diff()
    df.loc[:, 'delta_t'] = df[time_col].diff().fillna(0)
    first_time = df[time_col].iloc[0]
    df['time_since_start'] = df[time_col] - first_time
    
    return df


if __name__ == '__main__':
    print("开始测试 utils.py...")


    print("\n--- 测试时间解析 ---")
    time_str1 = "12:33:49.652093"
    time_str2 = "17:20:22.670443123" # 超过6位微秒
    time_str3 = "18:00:00"
    time_str4 = "invalid_time"
    
    print(f"'{time_str1}' -> {parse_time_to_seconds(time_str1):.6f} seconds")
    print(f"'{time_str2}' -> {parse_time_to_seconds(time_str2):.6f} seconds (微秒被截断)")
    print(f"'{time_str3}' -> {parse_time_to_seconds(time_str3):.6f} seconds")
    print(f"'{time_str4}' -> {parse_time_to_seconds(time_str4)}")


    print("\n--- 测试数据清洗 ---")
    data = {
        'A': [1, 2, np.nan, 4, '5'],
        'B': [1.1, 2.2, 3.3, np.nan, 'invalid'],
        'C': [np.nan, np.nan, np.nan, np.nan, np.nan] 
    }
    df = pd.DataFrame(data)
    print("原始DataFrame:")
    print(df)
    
    cleaned_df = clean_tabular_dataframe(df, numeric_cols=['A', 'B', 'C'])
    print("\n清洗后的DataFrame:")
    print(cleaned_df)
    print("\n清洗后数据类型:")
    print(cleaned_df.dtypes)
    