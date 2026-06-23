import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

# 1. 创建异构图数据
data = HeteroData()

# 添加用户节点
data['user'].x = torch.randn(4, 16)
data['user'].y = torch.tensor([0, 1, 0, 1])
data['user'].train_mask = torch.tensor([True, True, False, False])
data['user'].test_mask = torch.tensor([False, False, True, True])

# 添加商品节点
data['item'].x = torch.randn(5, 8)

# 添加边
data['user', 'rates', 'item'].edge_index = torch.tensor([
    [0, 0, 1, 1, 2, 3],
    [0, 1, 2, 3, 1, 4]
])
data['user', 'rates', 'item'].edge_attr = torch.tensor([5, 3, 4, 2, 5, 1], dtype=torch.float)

data['user', 'friends_with', 'user'].edge_index = torch.tensor([
    [0, 1, 2, 3],
    [1, 0, 3, 2]
])

# 添加反向边
data['item', 'rev_rates', 'user'] = data['user', 'rates', 'item'].edge_index.flip([0])

# 2. 定义模型
class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'rates', 'item'): SAGEConv((-1, -1), hidden_channels),
                ('item', 'rev_rates', 'user'): SAGEConv((-1, -1), hidden_channels),
                ('user', 'friends_with', 'user'): SAGEConv((-1, -1), hidden_channels),
            })
            self.convs.append(conv)
        
        self.lin = Linear(hidden_channels, out_channels)
    
    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['user'])

# 3. 训练和测试函数
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out[data['user'].train_mask], data['user'].y[data['user'].train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    pred = out.argmax(dim=-1)
    accs = []
    for mask in [data['user'].train_mask, data['user'].test_mask]:
        accs.append(int((pred[mask] == data['user'].y[mask]).sum()) / int(mask.sum()))
    return accs

# 4. 训练模型
model = HeteroGNN(hidden_channels=32, out_channels=2, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("开始训练...")
for epoch in range(1, 101):
    loss = train(model, data, optimizer, criterion)
    train_acc, test_acc = test(model, data)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')