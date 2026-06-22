#设置随机种子保证结果的可复现性
import numpy as np
#导入python随机数
import random
import os
import torch

# 定义一个函数setup_seed()，用于设置随机种子以确保结果的可复现性。该函数接受一个参数seed，表示要设置的随机种子值。
def setup_seed(seed):
    #设置Numpy随机种子，确保Numpy生成的随机数序列一致
    np.random.seed(seed)
    #设置python内置随机数种子，保证python内置的随机函数生成的随机数一致
    random.seed(seed)
    #设置python哈希种子，避免不同运行环境下的哈希结果不同，影响随机数生成
    # os.environ 用于访问或设置PYTHONHASHSEED环境变量
    os.environ['PYTHONHASHSEED']=str(seed)
    #torch.manual_seed()设置pytorch随机种子且固定种子，使pythoch生成随机数序列可以重复
    torch.manual_seed(seed)
    #检查是否有可用的cuda设备(GPU)
    if torch.cuda.is_available():
        #设置cuda随机数种子，保证在gpu上随即操作可重复
        torch.cuda.manual_seed(seed)
        #为所有GPU设置随机种子
        torch.cuda.manual_seed_all(seed)
        #关闭cudnn自动寻找最优算法加速功能，保证结果可复现
        torch.backends.cudnn.benchmark=False
        #设置cudnn为确定性算法，确保每次运行结果一致
        torch.backends.cudnn.deterministic=True
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('CUDA is useful!!')
else:
    device = torch.device("cpu")
    print('CUDA is not useful!!')
#设置随机数种子
setup_seed(0)

from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# 定义一个数据集类，继承自torch.utils.data.Dataset
# 该类用于加载和处理功率数据，进行归一化，并提供数据的访问接口
class PowerData(Dataset):
    def __init__(self,csv_path,sequence_len):
        self.sequence_len=sequence_len
        self.data=pd.read_csv(csv_path)
        #对数据进行截断处理如果功率(kW)数值超过1500
        self.data["功率(kW)"]=np.minimum(self.data["功率(kW)"],1500)
        #实例化归一化的类
        self.scaler=MinMaxScaler(feature_range=(-1,1))
        # reshape(-1, 1)之后得到[samples, 1]
        self.data["power_normalized"] =self.scaler.fit_transform(self.data["功率(kW)"].values.reshape(-1, 1))

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)-self.sequence_len
    def __getitem__(self, idx):
        start_idx=idx
        end_idx = idx + self.sequence_len
        #获得特征和目标值
        feature=self.data["power_normalized"].values[start_idx:end_idx]
        target = self.data["power_normalized"].values[end_idx:end_idx + 1]
        return torch.tensor(feature,dtype=torch.float32),torch.tensor(target, dtype=torch.float32)
sequence_len = 20
power_dataset =PowerData("../dataset/A01.csv",sequence_len)
#计算比列并划分每个数据集的内容
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
train_size = int(train_ratio * len(power_dataset))
val_size = int(val_ratio * len(power_dataset))
test_size = test_ratio * len(power_dataset)
#进行切片分成训练集和验证集
from torch.utils.data import Subset
indices = list(range(len(power_dataset)))
train_dataset = Subset(power_dataset, indices[:train_size])
val_dataset = Subset(power_dataset, indices[train_size:train_size + val_size])
test_dataset = Subset(power_dataset, indices[train_size + val_size:])

#加载和批量处理数据
from torch.utils.data  import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


import torch.nn as nn
class RNN(nn.Module):
    def __init__(self,input_size=1, hidden_size=128, output_size=1):
        """
        初始化
        :param input_size: 输入数据的特征维度大小，每个时间步输入的特征向量的维度
        :param hidden_size:
        :param output_size:
        """
        super(RNN,self).__init__()
        #定义rnn
        self.rnn=nn.RNN(input_size, hidden_size, batch_first=True)
        #定义线性变换
        self.linear=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x=x.unsqueeze(2)#转换为batch_size sequence_len input_size
        w,h_n=self.rnn(x)#h_n是最后一个时间步的隐藏状态
        x = (self.linear(h_n[-1])).squeeze(1)  # self.linear的输出张量[64, 1]
        return x

model=RNN().to(device)
cri=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-5)

epochs=20
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for batch_feature, batch_target in train_dataloader:
        batch_feature, batch_target = batch_feature.to(device), batch_target.to(device)
        # 前向传播
        y_pred = model(batch_feature)
        # y_pred [bs, 1] -> [bs,]
        loss = cri(y_pred, batch_target.view(-1))
        # 反向传播更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    train_loss = total_loss / len(train_dataloader)
    # 验证集的损失
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_feature, batch_target in val_dataloader:
            batch_feature, batch_target = batch_feature.to(device), batch_target.to(device)
            y_pred = model(batch_feature)
            total_loss += cri(y_pred, batch_target.view(-1)).item()
    val_loss = total_loss / len(val_dataloader)
    print(f"Epoch:[{epoch}/{epochs}], Train Loss: {train_loss:.4f}, Eval Loss: {val_loss:.4f}")

# 计算测试结果
model.eval()
predict_list = []
target_list = []
with torch.no_grad():
    for batch_feature, batch_target in test_dataloader:
        batch_feature, batch_target = batch_feature.to(device), batch_target.to(device)
        y_pred = model(batch_feature)
        predict_list.append(y_pred.item())
        target_list.append(batch_target.item())

# 将预测结果反归一化
# power_dataset.scaler.inverse_transform()：使用之前拟合的缩放器（scaler）对数据进行反归一化
predict_list = power_dataset.scaler.inverse_transform(np.array(predict_list).reshape(-1, 1))
# power_dataset.scaler.inverse_transform()：使用之前拟合的缩放器（scaler）对数据进行反归一化
target_list = power_dataset.scaler.inverse_transform(np.array(target_list).reshape(-1, 1))


import matplotlib.pyplot as plt
plt.plot(target_list, label="True values")
plt.plot(predict_list, label="Predict values")
plt.xlabel("Time")
plt.ylabel("Power")
plt.legend()
plt.show()