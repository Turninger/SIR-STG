import json
from collections import OrderedDict
"""
合并两个json文件并去重
"""
# 读取第一个文件
with open('../industry_keywords.json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)

# 读取第二个文件
with open('../industry_keywords1015.json', 'r', encoding='utf-8') as f:
    data2 = json.load(f)

# 创建合并后的有序字典
merged_data = OrderedDict()

# 使用第一个文件的行业顺序作为基础
for industry in data1.keys():
    keywords = set(data1[industry])
    if industry in data2:
        keywords.update(data2[industry])
    merged_data[industry] = sorted(list(keywords))

# 添加第二个文件中独有的行业
for industry in data2.keys():
    if industry not in merged_data:
        merged_data[industry] = sorted(data2[industry])

# 保存合并后的文件
with open('industry_keywords_merged.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"合并完成！共包含 {len(merged_data)} 个行业")