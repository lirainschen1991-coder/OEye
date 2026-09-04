import pandas as pd
import numpy as np
import re

def detect_file_format(file_path):
    """
    检测文件格式并返回数据起始行、列名行、单位行
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except PermissionError:
        raise PermissionError(f"没有权限访问文件: {file_path}")
    except Exception as e:
        raise IOError(f"读取文件时发生错误: {file_path}, 错误信息: {str(e)}")
    
    if not lines:
        raise ValueError(f"文件为空: {file_path}")
    
    # 先找到第一个数据行
    first_data_line = None
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#') or stripped_line.startswith('//'):
            continue
        
        # 检查是否包含数字（数据行）
        has_numbers = any(char.isdigit() or char in '.-+eE' for char in stripped_line)
        if has_numbers:
            # 尝试解析为数值列表
            values = re.split(r'\s+', stripped_line)
            values = [val for val in values if val]
            
            # 检查是否至少有3个数值（通常数据行有多个列）
            numeric_count = 0
            for val in values:
                try:
                    float(val)
                    numeric_count += 1
                except ValueError:
                    continue
            
            if numeric_count >= 3:
                first_data_line = i
                break
    
    if first_data_line is None:
        # 没有找到数据行
        raise ValueError(f"无法在文件中找到有效的数据行: {file_path}")
    
    # 现在寻找列名行（数据行之前的行）
    column_line = None
    unit_line = None
    
    # 检查数据行的前一行是否为单位行
    if first_data_line > 0:
        prev_line = lines[first_data_line - 1].strip()
        # 单位行通常包含括号或特殊字符
        if any(char in '()[]{}<>/' for char in prev_line):
            unit_line = first_data_line - 1
            # 单位行之前的行可能是列名行
            if first_data_line > 1:
                column_line = first_data_line - 2
                data_start = first_data_line
            else:
                column_line = None
                data_start = first_data_line
        else:
            # 数据行之前的行可能是列名行
            column_line = first_data_line - 1
            unit_line = None
            data_start = first_data_line
    else:
        # 数据行是第一行
        column_line = None
        unit_line = None
        data_start = 0
    
    # 增强检测：检查是否有常见的列名模式
    if column_line is not None:
        column_names = re.split(r'\s+', lines[column_line].strip())
        column_names = [col for col in column_names if col]
        if len(column_names) > 0:
            # 检查是否包含常见的列名
            common_columns = ['Time', 'time', 'TIME', 'X', 'Y', 'Z', 'Velocity', 'Acceleration', 'Force']
            if any(col in column_names for col in common_columns):
                # 确认这是列名行
                pass
    
    return column_line, unit_line, data_start

def read_data_file(file_path):
    """
    读取数据文件，支持.out、.dat、.txt、.csv、.xlsx和.xls格式
    """
    import os
    
    # 获取文件扩展名
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext in ['.out', '.dat', '.txt']:
            # 检测文件格式
            column_line, unit_line, data_start = detect_file_format(file_path)
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # 读取列名
            if column_line is not None:
                columns = re.split(r'\s+', lines[column_line].strip())
                # 移除空字符串
                columns = [col for col in columns if col]
                if not columns:
                    # 如果列名为空，使用默认命名
                    raise ValueError("检测到的列名为空")
            else:
                # 没有列名，使用默认命名
                # 先读取一行数据来确定列数
                num_columns = None
                for i in range(data_start, min(data_start + 10, len(lines))):  # 最多检查10行
                    data_line = re.split(r'\s+', lines[i].strip())
                    data_line = [col for col in data_line if col]
                    if data_line:
                        num_columns = len(data_line)
                        break
                
                if num_columns is None:
                    raise ValueError(f"无法从文件中读取数据列数: {file_path}")
                
                columns = [f'col_{i}' for i in range(num_columns)]
            
            # 读取数据
            data = []
            skipped_lines = 0
            for i, line in enumerate(lines[data_start:], start=data_start + 1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                
                # 使用正则表达式分割空白字符
                values = re.split(r'\s+', stripped_line)
                # 移除空字符串
                values = [val for val in values if val]
                
                # 处理值的数量与列数不匹配的情况
                if len(values) < len(columns):
                    # 如果值的数量少于列数，填充NaN
                    values += [None] * (len(columns) - len(values))
                    data.append(values)
                    skipped_lines += 1
                elif len(values) > len(columns):
                    # 如果值的数量多于列数，截断到列数
                    values = values[:len(columns)]
                    data.append(values)
                    skipped_lines += 1
                else:
                    data.append(values)
            
            if not data:
                raise ValueError(f"文件 {file_path} 中没有有效数据")
            
            if skipped_lines > 0:
                print(f"警告: 在文件 {file_path} 中，有 {skipped_lines} 行数据格式与列数不匹配，已进行自动处理")
            
            # 转换为DataFrame
            df = pd.DataFrame(data, columns=columns)
            
            # 将数值列转换为数值类型
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # 检查转换后是否全为NaN
                    if df[col].isnull().all():
                        # 如果全为NaN，保留原始字符串类型
                        df[col] = df[col].astype(str)
                except Exception as e:
                    # 如果转换失败，保留原始类型
                    df[col] = df[col].astype(str)
        
        elif file_ext == '.csv':
            # 读取CSV文件
            df = pd.read_csv(file_path)
        
        elif file_ext in ['.xlsx', '.xls']:
            # 读取Excel文件
            df = pd.read_excel(file_path)
        
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
    
    except Exception as e:
        raise ValueError(f"读取文件 {file_path} 时发生错误: {str(e)}")
    
    # 检查第一列是否为时间列
    if 'Time' in df.columns or 'time' in df.columns or 'TIME' in df.columns:
        time_col = [col for col in df.columns if col.lower() == 'time'][0]
        df = df.rename(columns={time_col: 'Time'})
    elif df.columns[0] not in ['Time', 'time', 'TIME'] and np.issubdtype(df.iloc[:, 0].dtype, np.number):
        # 如果第一列是数值列且不是明确的时间列名，假设它是时间列
        df = df.rename(columns={df.columns[0]: 'Time'})
    
    return df

def load_and_prepare_data(file_path, target_columns, feature_columns=None):
    """
    加载数据并根据目标列和特征列准备数据
    """
    # 读取数据
    df = read_data_file(file_path)
    
    # 如果没有指定特征列，使用除目标列外的所有列
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in target_columns]
    
    # 确保所有指定的列都存在
    for col in target_columns + feature_columns:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 不在数据中")
    
    # 提取特征和目标
    X = df[feature_columns]
    y = df[target_columns]
    
    return X, y, df


def read_multiple_files(file_paths, merge_method='concat', sort_by_time=True, remove_duplicates=True):
    """
    读取多个数据文件并合并
    
    参数:
        file_paths: 文件路径列表
        merge_method: 合并方法，'concat'表示纵向拼接，'join'表示横向合并
        sort_by_time: 是否按时间列排序
        remove_duplicates: 是否删除重复行
        
    返回:
        merged_df: 合并后的DataFrame
        file_info: 每个文件的信息列表
    """
    if not file_paths:
        raise ValueError("文件路径列表不能为空")
    
    dataframes = []
    file_info = []
    
    for i, file_path in enumerate(file_paths):
        try:
            df = read_data_file(file_path)
            
            # 添加文件来源列
            df['file_source'] = f"file_{i+1}"
            
            file_info.append({
                'index': i + 1,
                'path': file_path,
                'rows': len(df),
                'columns': list(df.columns)
            })
            
            dataframes.append(df)
            
        except Exception as e:
            file_info.append({
                'index': i + 1,
                'path': file_path,
                'error': str(e)
            })
    
    if not dataframes:
        raise ValueError("没有成功读取任何文件")
    
    # 合并数据
    if merge_method == 'concat':
        # 纵向拼接 - 需要确保列名一致
        # 获取所有文件的共同列
        common_columns = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_columns = common_columns.intersection(set(df.columns))
        common_columns = list(common_columns)
        
        if len(common_columns) < 2:  # 至少需要Time列和一列数据
            raise ValueError("文件之间没有足够的共同列进行合并")
        
        # 只保留共同列
        dataframes = [df[common_columns] for df in dataframes]
        
        # 纵向拼接
        merged_df = pd.concat(dataframes, ignore_index=True)
        
    elif merge_method == 'join':
        # 横向合并 - 基于时间列
        if 'Time' not in dataframes[0].columns:
            raise ValueError("横向合并需要Time列")
        
        merged_df = dataframes[0]
        for i, df in enumerate(dataframes[1:], start=2):
            if 'Time' not in df.columns:
                continue
            # 为每个文件的列添加前缀以避免冲突
            df_renamed = df.rename(columns={
                col: f"{col}_file{i}" if col != 'Time' else col 
                for col in df.columns
            })
            merged_df = pd.merge(merged_df, df_renamed, on='Time', how='outer')
    else:
        raise ValueError(f"不支持的合并方法: {merge_method}")
    
    # 按时间排序
    if sort_by_time and 'Time' in merged_df.columns:
        merged_df = merged_df.sort_values('Time').reset_index(drop=True)
    
    # 删除重复行
    if remove_duplicates and 'Time' in merged_df.columns:
        # 基于时间列删除重复行，保留第一次出现的
        merged_df = merged_df.drop_duplicates(subset=['Time'], keep='first')
    
    return merged_df, file_info


def batch_load_data(file_paths, target_columns, feature_columns=None, merge_method='concat'):
    """
    批量加载多个数据文件并准备训练数据
    
    参数:
        file_paths: 文件路径列表
        target_columns: 目标列列表
        feature_columns: 特征列列表，如果为None则使用除目标列外的所有列
        merge_method: 合并方法
        
    返回:
        X: 特征数据
        y: 目标数据
        df: 合并后的完整数据
        file_info: 文件信息
    """
    # 读取并合并多个文件
    df, file_info = read_multiple_files(file_paths, merge_method=merge_method)
    
    # 如果没有指定特征列，使用除目标列外的所有列
    if feature_columns is None:
        feature_columns = [col for col in df.columns 
                          if col not in target_columns and col != 'file_source']
    
    # 确保所有指定的列都存在
    for col in target_columns + feature_columns:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 不在合并后的数据中")
    
    # 删除包含NaN的行
    df_clean = df[feature_columns + target_columns].dropna()
    
    # 提取特征和目标
    X = df_clean[feature_columns]
    y = df_clean[target_columns]
    
    return X, y, df, file_info
