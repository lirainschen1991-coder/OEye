from src.data.data_loader import read_data_file

# 测试读取primary.out文件
df = read_data_file('primary.out')

print("数据读取成功！")
print(f"数据形状: {df.shape}")
print("列名:", df.columns.tolist())
print("\n前5行数据:")
print(df.head())
print("\n数据类型:")
print(df.dtypes)
