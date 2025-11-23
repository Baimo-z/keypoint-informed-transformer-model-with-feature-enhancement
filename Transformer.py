import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class AdaptiveFeatureSelector(nn.Module):
    def __init__(self, input_dim, num_factors=4, factor_dim=8, hidden_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.num_factors = num_factors
        self.factor_dim = factor_dim

        # 可学习因子 Q
        self.Q = nn.Parameter(torch.randn(num_factors, factor_dim))

        # Self-attention on Q
        self.self_attn = nn.MultiheadAttention(embed_dim=factor_dim, num_heads=2, batch_first=True)

        # Linear projection for input X
        self.input_proj = nn.Linear(input_dim, factor_dim)

        # Cross-attention: Q as Query, X as Key/Value
        self.cross_attn = nn.MultiheadAttention(embed_dim=factor_dim, num_heads=2, batch_first=True)

        # 回归器
        self.regressor = nn.Sequential(
            nn.Linear(num_factors * factor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        # 初始化可学习因子 Q
        nn.init.xavier_uniform_(self.Q)

        # 初始化 input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # 初始化 self_attn 和 cross_attn 的输出投影
        for attn in [self.self_attn, self.cross_attn]:
            nn.init.xavier_uniform_(attn.out_proj.weight)
            nn.init.zeros_(attn.out_proj.bias)

        # 初始化回归器 Linear 层
        for layer in self.regressor:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        batch_size = x.size(0)

        # Expand Q for batch
        Q = self.Q.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, num_factors, factor_dim)

        # Self-Attention on Q
        Q_updated, _ = self.self_attn(Q, Q, Q)

        # Project input to same dim
        x_seq = x  # self.input_proj(x)  # (B, factor_dim)

        # Cross-attention: Q attends to X
        Q_cross, _ = self.cross_attn(Q_updated, x_seq, x_seq)  # (B, num_factors, factor_dim)
        # Q_updated, _ = self.self_attn(Q_updated, Q_updated, Q_updated)
        # Q_cross, _ = self.cross_attn(Q_updated, x_seq, x_seq)
        # 点乘相似度（对每个输入特征做注意力加权）
        similarity = torch.bmm(Q_cross, x.transpose(1, 2))  # (B, num_factors)

        # Softmax 得到权重
        attn_weights = torch.softmax(similarity, dim=-1)  # (B, num_factors)

        # 加权融合 Q 表征
        final_factors = torch.bmm(attn_weights, x_seq)  # (B, factor_dim)
        fused = final_factors.reshape(final_factors.shape[0], -1)
        # 回归预测
        output = self.regressor(fused)  # (B, 1)
        return output, similarity


# 1. 加载数据并标准化
origin = pd.read_excel('./fin_data.xlsx')  # 假设这是上传的路径
label_0 = np.array(origin[['CD_sum']])
######进行minmax标准化
# 对label进行MinMax标准化
label_scaler = MinMaxScaler()
label = label_scaler.fit_transform(label_0)

# data_path = "/ssd/zzr/zch/new_data"
data_path = "./data_yasuo"

# 创建保存similarity的目录
save_dir = r"./similarity_CD"

csv_files = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))
print(csv_files)
all_data = []
for file_path in csv_files:
    X = pd.read_csv(file_path)
    X = np.array(X)
    all_data.append(X[:, np.newaxis, :])

# 纵向拼接：所有样本在一起
X = np.concatenate(all_data, axis=1)
print("输入数据的形状 (样本数, 时间步长, 特征数):", X.shape)

lower = np.percentile(X, 5, axis=0, keepdims=True)
upper = np.percentile(X, 100 - 5, axis=0, keepdims=True)
X_clipped = np.clip(X, lower, upper)

# Step 2: 归一化
mean = X_clipped.mean(axis=0, keepdims=True)
std = X_clipped.std(axis=0, keepdims=True)
std_adj = np.where(std < 1e-8, 1.0, std)

X_scaled = (X_clipped - mean) / std_adj

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, label, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. 初始化模型和优化器
input_dim = X.shape[2]
model = AdaptiveFeatureSelector(input_dim=input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

evaluation_done = False
# 5. 训练模型
for epoch in range(1000):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        preds, _ = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    # 验证
    if epoch % 2 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            y_pred, similarity = model(X_test_tensor)
            y_true = y_test_tensor
            # 计算评估指标
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            print(f"Epoch {epoch}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val MSE:   {mse:.4f}")
            print(f"  Val RMSE:  {rmse:.4f}")
            print(f"  Val MAE:   {mae:.4f}")
            print(f"  Val R2:    {r2:.4f}")
            # 把similarity转换成numpy后储存下来，保存成np的形式，名字以epoch来区分,base路径是D:\PythonProject\机器学习\similarity_CI
            # 保存similarity为numpy文件

            similarity_np = similarity.numpy()
            np.save(os.path.join(save_dir, f"similarity_epoch_{epoch}.npy"), similarity_np)

