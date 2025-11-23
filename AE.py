import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.preprocessing import MinMaxScaler

def safe_columnwise_standardize(X):
    """
    对每一列进行标准化：Z-score 标准化（均值为0，标准差为1），避免产生NaN。

    参数：
    - X: np.ndarray 或 pd.DataFrame，二维数组，shape=(samples, features)

    返回：
    - X_scaled: np.ndarray, 标准化后的数组，所有值 finite
    """

    X = X.copy()

    if isinstance(X, pd.DataFrame):
        X_np = X.values
    else:
        X_np = X

    mean = np.mean(X_np, axis=0)
    std = np.std(X_np, axis=0)

    # 避免除以 0（std = 0 的列，统一输出为 0）
    std_adj = np.where(std < 1e-8, 1.0, std)
    X_scaled = (X_np - mean) / std_adj

    # 对于 std=0 的列手动置为 0
    X_scaled[:, std < 1e-8] = 0.0

    return X_scaled

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=10):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x, flag='train'):
        encoded = self.encoder(x)
        if flag != 'train':
            return encoded
        decoded = self.decoder(encoded)
        return decoded
    
    
# ===========================================
# 1. 加载并标准化数据
# ===========================================
data_path = "./new_data"
save_path = "./data_yasuo"

csv_files = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

for file_path in csv_files:
    best_val_loss = float('inf')
    best_model_state = None
    
    X = pd.read_csv(file_path)
    X = np.array(X,dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    lower = np.percentile(X, 5, axis=0, keepdims=True)
    upper = np.percentile(X, 100 - 5, axis=0, keepdims=True)
    X_clipped = np.clip(X, lower, upper)

    # Step 2: 归一化
    mean = X_clipped.mean(axis=0, keepdims=True)
    std = X_clipped.std(axis=0, keepdims=True)
    std_adj = np.where(std < 1e-8, 1.0, std)

    X_scaled = (X_clipped - mean) / std_adj
    # ===========================================
    # 2. 划分训练集和测试集
    # ===========================================
    X_train, X_test = train_test_split(X_scaled, test_size=0.1, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # ===========================================
    # 4. 初始化训练配置
    # ===========================================
    input_dim = X.shape[1]
    model = AutoEncoder(input_dim=input_dim, bottleneck_dim=8)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 64
    num_epochs = 500

    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ===========================================
    # 5. 模型训练 (Mini-batch)
    # ===========================================
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            batch_x = batch[0]
            output = model(batch_x)
            loss = criterion(output, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        
        epoch_loss /= len(train_loader.dataset)
        
        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_output = model(X_test_tensor)
            val_loss = criterion(val_output, X_test_tensor)
            # 计算当前保真度（直接内联计算，不新增函数）
            val_mse = val_loss.item()
            val_fidelity = (1 - val_mse) * 100
        # 保存最优模型参数
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_fidelity = val_fidelity  # 同步记录保真度
            best_model_state = model.state_dict()  # 保存权重
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss.item():.4f}")

    # ===========================================
    # 6. 计算特征级重构误差（特征重要性）
    # ===========================================
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        X_pred = model(X_tensor,flag='test').numpy()
    # 获取原始文件名（带 .csv 后缀）
    file_name = os.path.basename(file_path)

    # 拼接保存路径
    output_file = os.path.join(save_path, file_name)

    # 转换为 DataFrame 并保存为 CSV
    df = pd.DataFrame(X_pred)
    df.to_csv(output_file, index=False)

    print(f"✅ Saved prediction to: {output_file}")
    print(f"\n🏆 最优模型指标:")
    print(f"- 验证集MSE: {best_val_loss:.6f}")
    print(f"- 对应保真度: {best_fidelity:.2f}%")

    # reconstruction_error = np.mean((X_test - X_test_pred) ** 2, axis=0)
    # important_features = pd.Series(reconstruction_error, index=X.columns).sort_values(ascending=False)

    # print("\nTop important features (by reconstruction error on test set):")
    # print(important_features.head(10))