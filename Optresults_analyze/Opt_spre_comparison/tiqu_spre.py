import re

# ------------ 用户可修改区域 ------------
input_file = "pltflw11P10MIN.plt"  # 输入文件名
output_file = "spre11P10MIN.txt"    # 输出文件名
# ----------------------------------------

# 读取文件并解析
with open(input_file, "r") as infile:
    lines = infile.readlines()

# 从第三行解析 I 的值（示例格式："ZONE T=' 1', I= 282, J= 30, F=POINT"）
third_line = lines[2].strip()
i_match = re.search(r"I=\s*(\d+)", third_line)  # 正则提取I值
if not i_match:
    raise ValueError(f"在第三行未找到I值: {third_line}")
I = int(i_match.group(1))

# 提取第4行到第I+3行的数据（共I行）
data_lines = lines[3 : 3 + I]

# 提取第1列和第7列数据
results = []
for line in data_lines:
    cols = line.strip().split()
    if len(cols) >= 7:
        # 保留原始数据格式（如科学计数法、小数点位数）
        results.append(f"{cols[0]} {cols[6]}")  # 第1列和第7列

# 写入输出文件
with open(output_file, "w") as outfile:
    outfile.write("\n".join(results))

print(f"成功提取 {I} 行数据到 {output_file}")