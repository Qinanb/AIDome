import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='logs')

'''
tensorboard --logdir="D:\pydome\AI25-12\Machine Learning\Day4\logs"
'''
# 1.散点输入
path='./dataset/ev_range_dataset.xlsx'

data = pd.read_excel(path).to_numpy()
# 提取x和y
x_data = data[:, 0]
y_data = data[:, 1]

# 把x，y转换成tensor
x_tarin = torch.tensor(x_data, dtype=torch.float32)
y_tarin = torch.tensor(y_data, dtype=torch.float32)


# 用于封装张量，将输入张量和输出张量组成一个数据集
# 返回值能够按照索引获得数据和标签，例如(x_train[i],y_trian[i])
dataset=TensorDataset(x_tarin,y_tarin)


# 设置随机数种子
seed = 32
torch.manual_seed(seed)

# 方案2: 手动编写,直接重写继承nn.Module

class LinearModel(torch.nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super(LinearModel, self).__init__()
        self.layer1 = torch.nn.Linear(1, 1)
        # self.layer2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        return x


# # 创建模型对象
model = LinearModel()
# 3.定义损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)
# 4.开始迭代
epoches = 500

# dataloader可迭代对象，每次迭代会产生一个batch的数据，由输入张量和目标张量元组 组成
dataloader=DataLoader(dataset,
           batch_size=5,
           shuffle=True,
           )

fig, ax = plt.subplots(figsize=(8, 6))  # 只创建一次画布和轴对象
scatter = ax.scatter(x_data, y_data, color='blue', label='Original Data', alpha=0.6)  # 原始散点只画一次
line, = ax.plot([], [], color='red', label='Fitted Line')  # 初始化拟合线（空数据）


ax.set_xlabel('X ')
ax.set_ylabel('Y ')
ax.set_title('Linear Regression Fitting Result')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(min(x_data), max(x_data))
ax.set_ylim(min(y_data), max(y_data))

# 生成拟合线的x坐标（固定，无需重复生成）
x_fit = np.linspace(min(x_data), max(x_data), 100)

for epoch in range(1, epoches + 1):
    total_loss = 0
    for x_batch, y_batch in dataloader:
        y_hat = model(x_batch.unsqueeze(1))
        loss = criterion(y_hat, y_batch.unsqueeze(1))
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 计算平均损失
    avg_loss = total_loss / len(dataloader)

    # 5.显示频率设置（按epoch更新图像）
    if epoch == 1 or epoch % 10 == 0:
        w = model.layer1.weight.item()
        b = model.layer1.bias.item()
        print(f'epoch:{epoch},loss:{avg_loss}-----> y={w:.4f}x{b:.4f}')

        # 记录 loss 到 TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)


        # 计算拟合线的y坐标
        y_fit = w * x_fit + b

        line.set_xdata(x_fit)  # 更新拟合线X数据
        line.set_ydata(y_fit)  # 更新拟合线Y数据
        line.set_label(f'Fitted Line: y={w:.4f}x{b:.4f}')  # 更新标签
        ax.legend()  # 刷新图例
        fig.canvas.draw()  # 重绘画布
        plt.pause(0.1)

writer.close()
plt.show()



