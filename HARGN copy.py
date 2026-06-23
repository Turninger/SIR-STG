import torch
import os
import pandas as pd
from torch_geometric.data import HeteroData
from datetime import datetime
import numpy as np
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import Dropout
from torch_geometric.nn import HANConv, Linear  # 使用HANConv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
from n_day_HeteroGraphSL_copy import build_or_load_dataset

# # 构建或加载数据集
# train_graphs, test_graphs = build_or_load_dataset()

# # # 从训练集中划分验证集
# # train_graphs, val_graphs = train_test_split(
# #     train_graphs, 
# #     test_size=0.2, 
# #     random_state=42
# # )

# # 只保留前1/2的图 (按时间顺序)
# total_graphs = train_graphs + test_graphs
# half_len = len(total_graphs) // 2
# total_graphs = total_graphs[:half_len]  # 只取前一半

# # 重新划分训练集和测试集 (按时间顺序)
# train_size = int(0.8 * len(total_graphs))
# train_graphs = total_graphs[:train_size]
# test_graphs = total_graphs[train_size:]

# # 从训练集中按时间顺序划分验证集 (训练集的后20%)
# val_size = int(0.2 * len(train_graphs))
# val_graphs = train_graphs[-val_size:]
# train_graphs = train_graphs[:-val_size]

# # 创建数据加载器
# batch_size = 16
# train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)


from n_day_HeteroGraphSL import build_or_load_dataset

# 构建或加载数据集
train_graphs, test_graphs = build_or_load_dataset()

# 从训练集中划分验证集
train_graphs, val_graphs = train_test_split(
    train_graphs, 
    test_size=0.2, 
    random_state=42
)

# 创建数据加载器
batch_size = 16
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

print(f"数据集大小: 训练集={len(train_graphs)}, 验证集={len(val_graphs)}, 测试集={len(test_graphs)}")

class HANHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, heads=8,dropout_rate=0.5):
        super().__init__()
        # 从第一个图中获取元数据 - 直接访问属性
        self.metadata = train_graphs[0].metadata() 
    
        
        # 特征变换层 
        self.lin_dict = torch.nn.ModuleDict({
            'stock': Linear(8, hidden_channels),
            'news': Linear(1, hidden_channels),
            'industry': Linear(1, hidden_channels)
        })

        self.lin_dict2 = torch.nn.ModuleDict({
            'stock': Linear(8, 64),
            'news': Linear(1, 64),
            'industry': Linear(1, 64)
        })


        self.lin_dict3 = torch.nn.ModuleDict({
            'stock': Linear(8, 32),
            'news': Linear(1, 32),
            'industry': Linear(1, 32)
        })
        
        # 第一层HANConv: 输入hidden_channels -> 输出512/8=64
        self.conv1 = HANConv(
            in_channels=hidden_channels,
            out_channels=512 // heads,  # 确保输出维度为512
            heads=heads,
            dropout=dropout_rate,
            metadata=self.metadata
        )
        
        # 第二层HANConv: 输入64 -> 输出256/8=32
        self.conv2 = HANConv(
            in_channels=64,
            out_channels=256 // heads,  # 确保输出维度为256
            heads=heads,
            dropout=dropout_rate,
            metadata=self.metadata
        )
        
        # 第三层HANConv: 输入32 -> 输出64/8=8
        self.conv3 = HANConv(
            in_channels=32,
            out_channels=64 // heads,  # 确保输出维度为64
            heads=heads,
            dropout=dropout_rate,
            metadata=self.metadata
        )
        
        # 最终分类器 - 输入64维特征
        self.classifier = Linear(8, out_channels)
        
        # Dropout层
        self.dropout = Dropout(dropout_rate)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_dict, edge_index_dict):
        # 特征变换 + ReLU + Dropout
        # 保存初始特征（用于非目标节点）
        initial_news = x_dict['news'].clone()
        initial_industry = x_dict['industry'].clone()

        x_dict = {
            'stock': F.relu(self.dropout(self.lin_dict['stock'](x_dict['stock']))),
            'news': F.relu(self.dropout(self.lin_dict['news'](x_dict['news']))),
            'industry': F.relu(self.dropout(self.lin_dict['industry'](x_dict['industry'])))
        }

        # 第一层HANConv + ReLU
        x_dict = self.conv1(x_dict, edge_index_dict)
        stock_features = F.relu(self.dropout(x_dict['stock']))
        #x_dict = {key: F.relu(value) for key, value in x_dict.items()}
        #x_dict['stock'] = F.relu(self.dropout(x_dict['stock']))

        # x_dict = {
        # 'stock': stock_features,
        # 'news': F.relu(self.lin_dict['news'](initial_news)),
        # 'industry': F.relu(self.lin_dict['industry'](initial_industry))
        # }

        x_dict = {
        'stock': stock_features,
        'news': F.relu(self.dropout(self.lin_dict2['news'](initial_news))),
        'industry': F.relu(self.dropout(self.lin_dict2['industry'](initial_industry)))
        }
   
        
        # 第二层HANConv + ReLU
        x_dict = self.conv2(x_dict, edge_index_dict)
        stock_features = F.relu(self.dropout(x_dict['stock']))
        #x_dict['stock'] = F.relu(self.dropout(x_dict['stock']))
        #x_dict = {key: F.relu(value) for key, value in x_dict.items()}
        #x_dict = {key: self.dropout(value) for key, value in x_dict.items()}

        # x_dict = {
        # 'stock': stock_features,
        # 'news': F.relu(self.lin_dict['news'](initial_news)),
        # 'industry': F.relu(self.lin_dict['industry'](initial_industry))
        # }
        
        x_dict = {
        'stock': stock_features,
        'news': F.relu(self.dropout(self.lin_dict3['news'](initial_news))),
        'industry': F.relu(self.dropout(self.lin_dict3['industry'](initial_industry)))
        }
        
        # 第三层HANConv + ReLU
        x_dict = self.conv3(x_dict, edge_index_dict)
        stock_features = F.relu(self.dropout(x_dict['stock']))
        #x_dict['stock'] = F.relu(self.dropout(x_dict['stock']))
        #x_dict = {key: F.relu(value) for key, value in x_dict.items()}
        #x_dict = {key: self.dropout(value) for key, value in x_dict.items()}
        
        # 只返回股票节点的预测结果
        stock_features = x_dict['stock']
        stock_features = self.sigmoid(stock_features)
        
        return self.classifier(stock_features)


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        y = batch['stock'].y
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
    return total_loss / total_samples

def evaluate(model, loader, criterion, device, return_predictions=False):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_indices = []  # 存储批次索引
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            y = batch['stock'].y
            loss = criterion(out, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_indices.extend([batch_idx] * y.size(0))  # 记录批次索引
    
    # 计算多种指标
    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    mcc = matthews_corrcoef(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': acc,
        'f1': f1,
        'recall': recall,
        'mcc': mcc,
        'auc': auc
    }
    
    if return_predictions:
        return metrics, all_preds, all_labels, all_probs, all_indices
    return metrics

def plot_training_history(history, file_name="./result/5_day_HAN_result/training_history.png"):
    """可视化训练过程"""
    plt.figure(figsize=(15, 10))
    
    # 1. 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['epochs'], history['train_loss'], 'b-', label='Train Loss')
    plt.plot(history['epochs'], history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['epochs'], history['val_accuracy'], 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 3. AUC曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['epochs'], history['val_auc'], 'm-', label='Validation AUC')
    plt.title('Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    # 4. F1分数曲线
    plt.subplot(2, 2, 4)
    plt.plot(history['epochs'], history['val_f1'], 'c-', label='Validation F1')
    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(file_name)
    print(f"训练历史图表已保存为: {file_name}")
    plt.close()

def main():
    # 设置随机种子以确保可复现性
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化HAN模型
    model = HANHeteroGNN(
        hidden_channels=128,  # 初始特征维度
        out_channels=2,       # 分类类别数
        heads=8,
        dropout_rate=0.8
    ).to(device)
    
    # 打印模型结构
    print("模型结构:")
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 记录最佳模型
    best_val_auc = 0
    best_val_acc =0
    best_model_state = None
    best_epoch = 0
    
    # 初始化训练历史记录
    training_history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_recall': [],
        'val_mcc': [],
        'val_auc': []
    }
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(1, 300):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        
        # 在验证集上评估
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # 记录训练历史
        training_history['epochs'].append(epoch)
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_metrics['loss'])
        training_history['val_accuracy'].append(val_metrics['accuracy'])
        training_history['val_f1'].append(val_metrics['f1'])
        training_history['val_recall'].append(val_metrics['recall'])
        training_history['val_mcc'].append(val_metrics['mcc'])
        training_history['val_auc'].append(val_metrics['auc'])
        
        # 打印训练和验证结果
        print(f'\nEpoch: {epoch:03d}, Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')
        print(f'Val F1: {val_metrics["f1"]:.4f}, Val Recall: {val_metrics["recall"]:.4f}')
        print(f'Val MCC: {val_metrics["mcc"]:.4f}, Val AUC: {val_metrics["auc"]:.4f}')
        
        # 保存最佳模型（基于验证集AUC）
        # if val_metrics['auc'] > best_val_auc:
        #     best_val_auc = val_metrics['auc']
        #     best_val_acc = val_metrics['acc']
        #     best_model_state = model.state_dict().copy()
        #     best_epoch = epoch
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': best_model_state,
        #         'val_auc': best_val_auc,
        #         'optimizer_state_dict': optimizer.state_dict(),
        #     }, 'best_han_model.pth')
        #     print(f"🌟 发现新最佳模型 (Val AUC={best_val_auc:.4f})")


        if val_metrics['accuracy'] > best_val_acc:
            best_val_auc = val_metrics['auc']
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'val_auc': best_val_auc,
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'best_han_model.pth')
            print(f"🌟 发现新最佳模型 (Val ACC={best_val_acc:.4f})")
        
        # 每50个epoch保存一次训练图表
        if epoch % 50 == 0:
            plot_training_history(training_history, f"./result/5_day_HAN_result/training_history_epoch_{epoch}.png")
        
        print('-' * 80)
    
    # 训练结束后绘制最终训练图表
    plot_training_history(training_history, "./result/5_day_HAN_result/final_training_history.png")
    
    # 加载最佳模型
    if best_model_state is not None:
        checkpoint = torch.load('best_han_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n加载第 {checkpoint['epoch']} 轮的最佳模型 (Val AUC={checkpoint['val_auc']:.4f})")
    
    # 在测试集上评估最终性能
    test_metrics, test_preds, test_labels, test_probs, test_indices = evaluate(
        model, test_loader, criterion, device, return_predictions=True)
    
    print("\n📊 最终测试结果:")
    print(f'Test Loss: {test_metrics["loss"]:.4f}, Test Acc: {test_metrics["accuracy"]:.4f}')
    print(f'Test F1: {test_metrics["f1"]:.4f}, Test Recall: {test_metrics["recall"]:.4f}')
    print(f'Test MCC: {test_metrics["mcc"]:.4f}, Test AUC: {test_metrics["auc"]:.4f}')
    
    # 保存测试结果
    results_df = pd.DataFrame({
        'batch_index': test_indices,
        'true_label': test_labels,
        'predicted_label': test_preds,
        'probability': test_probs
    })
    results_df.to_csv('./5_day_result/test_results.csv', index=False)
    print("测试结果已保存到 './5_day_result/test_results.csv'")
    
    # 保存完整模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_metrics': test_metrics,
    }, 'final_han_model.pth')
    print("完整模型已保存到 './result/5_day_HAN_result/final_han_model.pth'")

if __name__ == "__main__":
    main()