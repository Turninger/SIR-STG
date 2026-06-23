import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 创建数据框
data = {
    'Methods': ['ARIMA', 'LSTM', 'ALSTM', 'Transformer', 'Graph-SAGE', 'RGCN', 'GAT', 'HAN', 'HARGN'] * 3,
    'T': ['3-days'] * 9 + ['5-days'] * 9 + ['7-days'] * 9,
    'CSI100_ACC': [0.4367, 0.5053, 0.5214, 0.5329, 0.5039, 0.5459, 0.5134, 0.5607, 0.5803,
                   0.5053, 0.5112, 0.5094, 0.5253, 0.5257, 0.5393, 0.5604, 0.5409, 0.6021,
                   0.4886, 0.519, 0.5172, 0.5061, 0.5064, 0.5131, 0.5156, 0.517, 0.589],
    'CSI100_F1': [0.366, 0.6041, 0.592, 0.5937, 0.4724, 0.6721, 0.5396, 0.6807, 0.7135,
                  0.3689, 0.6638, 0.6479, 0.6663, 0.4061, 0.6605, 0.7512, 0.6773, 0.7514,
                  0.3303, 0.6385, 0.6746, 0.6704, 0.5832, 0.6545, 0.6604, 0.68, 0.6804],
    'CSI100_Recall': [0.3151, 0.5531, 0.4636, 0.6071, 0.5289, 0.5879, 0.2157, 0.793, 0.7451,
                      0.2905, 0.5761, 0.6416, 0.6356, 0.3911, 0.7445, 0.6872, 0.8004, 0.6992,
                      0.2535, 0.6523, 0.7431, 0.6767, 0.6918, 0.8945, 0.5631, 0.6728, 0.8874],
    'CSI100_MCC': [0.0121, 0.0518, 0.0359, -0.0069, 0.0145, 0.0261, 0.0026, 0.0224, 0.126,
                   0.0103, 0.1704, 0.0286, 0.0235, 0.0122, 0.0227, 0.004, 0.0655, 0.0066,
                   -0.0284, 0.0201, 0.0147, 0.0231, 0.0061, 0.0026, 0.003, 0.0217, 0.0015],
    'CSI300_ACC': [0.5244, 0.5267, 0.5216, 0.5202, 0.5094, 0.5179, 0.5189, 0.5454, 0.6045,
                   0.5048, 0.5234, 0.5289, 0.525, 0.5023, 0.5041, 0.5319, 0.5657, 0.6098,
                   0.4713, 0.5083, 0.5061, 0.5191, 0.5277, 0.5379, 0.5188, 0.5309, 0.6053],
    'CSI300_F1': [0.2163, 0.5372, 0.6083, 0.6263, 0.5416, 0.3767, 0.5665, 0.4634, 0.7576,
                  0.3725, 0.6277, 0.6291, 0.6608, 0.5274, 0.4907, 0.5987, 0.6156, 0.7527,
                  0.3957, 0.6611, 0.676, 0.6713, 0.482, 0.374, 0.5315, 0.6548, 0.6894],
    'CSI300_Recall': [0.3028, 0.478, 0.5951, 0.6308, 0.5878, 0.3175, 0.6784, 0.4677, 0.5363,
                      0.3264, 0.6274, 0.6289, 0.6455, 0.6639, 0.3954, 0.8677, 0.7319, 0.8394,
                      0.3411, 0.6516, 0.6896, 0.665, 0.4423, 0.266, 0.4811, 0.7048, 0.7945],
    'CSI300_MCC': [-0.009, 0.0233, 0.0162, 0.0249, 0.0213, 0.0054, 0.0057, 0.0064, 0.0108,
                   0.0023, 0.0293, 0.026, 0.0289, 0.0071, 0.0662, 0.0067, 0.0203, 0.039,
                   0.0004, 0.0235, 0.0308, 0.0314, 0.0552, 0.0309, 0.0015, 0.0221, 0.0556]
}

# data= pd.read_csv("result.csv")
df = pd.DataFrame(data)

# # 修复异常值（RGCN 5-days CSI100_ACC应该是0.39而不是39）
# df.loc[14, 'CSI100_ACC'] = 0.39

# 设置时间顺序
time_order = ['3-days', '5-days', '7-days']
df['T'] = pd.Categorical(df['T'], categories=time_order, ordered=True)

# 定义方法列表
methods = ['ARIMA', 'LSTM', 'ALSTM', 'Transformer', 'Graph-SAGE', 'RGCN', 'GAT', 'HAN', 'HARGN']

# 定义颜色映射
colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
color_dict = dict(zip(methods, colors))

# 创建趋势分析图表
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Analysis of indicator trends for different methods across various time spans', fontsize=16, fontweight='bold')

# CSI100指标趋势
metrics_csi100 = ['CSI100_ACC', 'CSI100_F1', 'CSI100_Recall', 'CSI100_MCC']
metric_names = ['ACC', 'F1', 'Recall', 'MCC']

for i, (metric, name) in enumerate(zip(metrics_csi100, metric_names)):
    ax = axes[0, i]
    for method in methods:
        method_data = df[df['Methods'] == method].sort_values('T')
        ax.plot(method_data['T'], method_data[metric], 
                marker='o', linewidth=2, label=method, color=color_dict[method])
    
    ax.set_title(f'CSI100 - {name}')
    ax.set_xlabel('时间跨度')
    ax.set_ylabel(name)
    ax.tick_params(axis='x', rotation=45)

# CSI300指标趋势
metrics_csi300 = ['CSI300_ACC', 'CSI300_F1', 'CSI300_Recall', 'CSI300_MCC']

for i, (metric, name) in enumerate(zip(metrics_csi300, metric_names)):
    ax = axes[1, i]
    for method in methods:
        method_data = df[df['Methods'] == method].sort_values('T')
        ax.plot(method_data['T'], method_data[metric], 
                marker='o', linewidth=2, label=method, color=color_dict[method])
    
    ax.set_title(f'CSI300 - {name}')
    ax.set_xlabel('time')
    ax.set_ylabel(name)
    ax.tick_params(axis='x', rotation=45)

# 添加图例
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
           ncol=5, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# 创建每个方法的单独趋势图
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('The comprehensive performance trend of various methods across different time spans', fontsize=16, fontweight='bold')

axes = axes.flatten()

for idx, method in enumerate(methods):
    ax = axes[idx]
    method_data = df[df['Methods'] == method].sort_values('T')
    
    # CSI100指标
    ax.plot(method_data['T'], method_data['CSI100_ACC'], marker='o', linewidth=2, 
            label='CSI100 ACC', color='blue', alpha=0.7)
    ax.plot(method_data['T'], method_data['CSI100_F1'], marker='s', linewidth=2, 
            label='CSI100 F1', color='green', alpha=0.7)
    ax.plot(method_data['T'], method_data['CSI100_Recall'], marker='^', linewidth=2, 
            label='CSI100 Recall', color='red', alpha=0.7)
    
    # CSI300指标
    ax.plot(method_data['T'], method_data['CSI300_ACC'], marker='o', linewidth=2, 
            label='CSI300 ACC', color='blue', linestyle='--', alpha=0.7)
    ax.plot(method_data['T'], method_data['CSI300_F1'], marker='s', linewidth=2, 
            label='CSI300 F1', color='green', linestyle='--', alpha=0.7)
    ax.plot(method_data['T'], method_data['CSI300_Recall'], marker='^', linewidth=2, 
            label='CSI300 Recall', color='red', linestyle='--', alpha=0.7)
    
    ax.set_title(f'{method}Method')
    ax.set_xlabel('Time')
    ax.set_ylabel('Indicator')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# 创建热力图显示性能变化
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# CSI100性能变化热力图
csi100_pivot = df.pivot(index='Methods', columns='T', values='CSI100_ACC')
sns.heatmap(csi100_pivot, annot=True, cmap='YlOrRd', ax=axes[0], 
            cbar_kws={'label': 'ACC'})
axes[0].set_title('Result on CSI100')

# CSI300性能变化热力图
csi300_pivot = df.pivot(index='Methods', columns='T', values='CSI300_ACC')
sns.heatmap(csi300_pivot, annot=True, cmap='YlOrRd', ax=axes[1], 
            cbar_kws={'label': 'ACC'})
axes[1].set_title('Result on CSI300')

plt.tight_layout()
plt.show()

# 输出关键观察结果
print("=" * 60)
print("关键趋势观察结果:")
print("=" * 60)

# 分析每个方法的表现趋势
for method in methods:
    method_data = df[df['Methods'] == method].sort_values('T')
    
    # CSI100 ACC趋势
    csi100_acc_trend = "上升" if method_data['CSI100_ACC'].iloc[-1] > method_data['CSI100_ACC'].iloc[0] else "下降"
    csi100_acc_change = method_data['CSI100_ACC'].iloc[-1] - method_data['CSI100_ACC'].iloc[0]
    
    # CSI300 ACC趋势
    csi300_acc_trend = "上升" if method_data['CSI300_ACC'].iloc[-1] > method_data['CSI300_ACC'].iloc[0] else "下降"
    csi300_acc_change = method_data['CSI300_ACC'].iloc[-1] - method_data['CSI300_ACC'].iloc[0]
    
    print(f"{method}:")
    print(f"  CSI100 ACC: {csi100_acc_trend}趋势 (变化: {csi100_acc_change:.4f})")
    print(f"  CSI300 ACC: {csi300_acc_trend}趋势 (变化: {csi300_acc_change:.4f})")
    print()

# 找出最佳表现的方法
best_methods = {}
for metric in ['CSI100_ACC', 'CSI100_F1', 'CSI100_Recall', 'CSI300_ACC', 'CSI300_F1', 'CSI300_Recall']:
    best_method = df.loc[df[metric].idxmax(), 'Methods']
    best_value = df[metric].max()
    best_methods[metric] = (best_method, best_value)

print("各指标最佳表现方法:")
for metric, (method, value) in best_methods.items():
    print(f"  {metric}: {method} ({value:.4f})")