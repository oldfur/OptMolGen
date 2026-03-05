import pandas as pd

# 读取Excel文件
file_path = './邻甲基苯醛.xlsx'
df = pd.read_excel(file_path)

# 打印前5行数据
print(df.head())
