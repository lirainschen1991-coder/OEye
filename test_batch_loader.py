from src.data.data_loader import read_multiple_files, batch_load_data
import os

# 测试批量读取功能
print("=" * 60)
print("测试批量数据读取功能")
print("=" * 60)

# 获取当前目录下的所有.out和.dat文件
data_files = []
for file in os.listdir('.'):
    if file.endswith(('.out', '.dat')):
        data_files.append(file)

print(f"\n找到的数据文件: {data_files}")

if len(data_files) >= 2:
    print("\n" + "=" * 60)
    print("测试1: 纵向拼接模式 (concat)")
    print("=" * 60)
    
    try:
        df_concat, file_info = read_multiple_files(
            data_files[:2], 
            merge_method='concat',
            sort_by_time=True,
            remove_duplicates=True
        )
        
        print(f"✓ 成功合并 {len(file_info)} 个文件")
        print(f"✓ 合并后数据形状: {df_concat.shape}")
        print(f"✓ 列名: {df_concat.columns.tolist()[:5]}...")  # 只显示前5列
        print(f"✓ 时间范围: {df_concat['Time'].min():.3f} - {df_concat['Time'].max():.3f}")
        
        # 显示每个文件的信息
        print("\n文件详情:")
        for info in file_info:
            if 'error' in info:
                print(f"  ✗ 文件 {info['index']}: {info['path']} - 错误: {info['error']}")
            else:
                print(f"  ✓ 文件 {info['index']}: {info['path']} ({info['rows']} 行)")
        
    except Exception as e:
        print(f"✗ 纵向拼接失败: {str(e)}")
    
    print("\n" + "=" * 60)
    print("测试2: 横向合并模式 (join)")
    print("=" * 60)
    
    try:
        df_join, file_info = read_multiple_files(
            data_files[:2], 
            merge_method='join',
            sort_by_time=True,
            remove_duplicates=True
        )
        
        print(f"✓ 成功合并 {len(file_info)} 个文件")
        print(f"✓ 合并后数据形状: {df_join.shape}")
        print(f"✓ 列数: {len(df_join.columns)}")
        print(f"✓ 时间范围: {df_join['Time'].min():.3f} - {df_join['Time'].max():.3f}")
        
    except Exception as e:
        print(f"✗ 横向合并失败: {str(e)}")
    
    print("\n" + "=" * 60)
    print("测试3: 批量加载数据 (batch_load_data)")
    print("=" * 60)
    
    try:
        # 假设第一个文件的第二列是目标列
        target_cols = [df_concat.columns[1]]
        
        X, y, df, info = batch_load_data(
            data_files[:2],
            target_columns=target_cols,
            merge_method='concat'
        )
        
        print(f"✓ 成功批量加载数据")
        print(f"✓ 特征矩阵形状: {X.shape}")
        print(f"✓ 目标变量形状: {y.shape}")
        print(f"✓ 特征列: {X.columns.tolist()[:3]}...")  # 只显示前3列
        print(f"✓ 目标列: {y.columns.tolist()}")
        
    except Exception as e:
        print(f"✗ 批量加载失败: {str(e)}")

else:
    print(f"\n✗ 需要至少2个数据文件进行批量测试，当前只有 {len(data_files)} 个")
    print("请确保目录下至少有两个.out或.dat文件")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
