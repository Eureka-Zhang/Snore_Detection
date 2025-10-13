# train_cnn_lstm.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================
# 数据加载部分
# =====================
class SnoreDataset(Dataset):
    def __init__(self, feature_dir):
        self.files = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith('.npy')]
        self.labels = [1 if 'snore' in f else 0 for f in self.files]  # 假设文件名包含 snore/non_snore

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # [1, freq, time]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label

# =====================
# CNN+LSTM 模型
# =====================
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(8, 2), padding=(4, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(16, 32, kernel_size=(8, 2), padding=(4, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(32, 64, kernel_size=(8, 2), padding=(4, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.5)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [B, 1, F, T]
        x = self.cnn(x)  # [B, 64, F', T']
        x = torch.mean(x, dim=2)  # 频率方向平均 -> [B, 64, T']
        x = x.permute(0, 2, 1)  # [B, T', 64]
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

# =====================
# 训练逻辑
# =====================
def train_model(feature_dir, epochs=20, batch_size=8, lr=1e-3):
    dataset = SnoreDataset(feature_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = CNN_LSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for data, label in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "snore_cnn_lstm.pth")
    print("Model saved as snore_cnn_lstm.pth")

if __name__ == "__main__":
    train_model("features/")
