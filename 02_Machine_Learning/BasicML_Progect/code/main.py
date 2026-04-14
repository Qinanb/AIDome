import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tensorboardX import SummaryWriter
from datetime import datetime

# 1.读取数据集
file_path = '../dataset/ENB2012_data.xlsx'
data = pd.read_excel(file_path)

# 打印前5行数据，检查是否读取成功
# print(data.head(5))

# 统计是否有空值
# print(data.isnull().sum())

# 2.处理数据集
# 对x6和x8进行独热编码
features_list = ['X6', 'X8']
data = pd.get_dummies(
    data,
    columns=features_list
)
print(data.keys())

# 提取特征和目标变量
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X7', 'X6_2', 'X6_3', 'X6_4',
          'X6_5', 'X8_0', 'X8_1', 'X8_2', 'X8_3', 'X8_4', 'X8_5']]
Y = data[['Y1', 'Y2']]

# 设置随机数种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
# 将数据随机划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# 数据标准化
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

# 将处理好的数据转换成tensor
x_train_tensor = torch.tensor(x_train_scaler, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test_scaler, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


# 生成带时间戳的日志文件夹（避免日志覆盖）
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'logs/{current_time}_energy_regression'  # 日志保存路径
writer = SummaryWriter(log_dir=log_dir)  # 创建日志写入器


# 3.构建模型
class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        return x


# 实例化模型
model = LinearRegressionModel(x_train_tensor.shape[1], y_train_tensor.shape[1])

# 写入模型结构到TensorBoard
# 创建示例输入（维度与真实输入一致）
dummy_input = torch.randn(1, x_train_tensor.shape[1], dtype=torch.float32)
writer.add_graph(model, dummy_input)  # 可视化模型结构

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

# 4.模型训练
epochs = 5000
for epoch in range(1,epochs + 1):
    # 模型设置为训练模式
    model.train()
    # 清空梯度
    optimizer.zero_grad()

    # 前向传播并得到损失
    output = model(x_train_tensor)
    loss = criterion(output, y_train_tensor)

    # 根据得到的损失进行反向传播
    loss.backward()
    # 参数更新

    optimizer.step()

    # 计算验证损失并写入TensorBoard
    model.eval()
    with torch.no_grad():
        val_output = model(x_test_tensor)
        val_loss = criterion(val_output, y_test_tensor)

    # 记录训练/验证损失（每轮都记录，TensorBoard可看完整曲线）
    writer.add_scalars(
        main_tag='Loss',
        tag_scalar_dict={
            'Train Loss': loss.item(),
            'Val Loss': val_loss.item()
        },
        global_step=epoch
    )

    # 打印损失值
    if epoch % 100 == 0 or epoch == 1:
        print(f'Epoch [{epoch}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

print('==================================================================\n')

# 评估模型
model.eval()
with torch.no_grad():
    predictions = model(x_test_tensor)
    y_pred = predictions.numpy()
    y_true = y_test_tensor.numpy()

# 拆分两个目标变量的预测结果和真实值
y1_pred, y2_pred = y_pred[:, 0], y_pred[:, 1]
y1_true, y2_true = y_true[:, 0], y_true[:, 1]


def calculate_metrics(y_true, y_pred, target_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'=== {target_name} 评估结果 ===')
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}\n")
    return rmse, r2


# 计算Y1和Y2的指标
rmse_y1, r2_y1 = calculate_metrics(y1_true, y1_pred, 'Y1:热负荷')
rmse_y2, r2_y2 = calculate_metrics(y2_true, y2_pred, 'Y2:冷负荷')

# 整体指标
avg_rmse = (rmse_y1 + rmse_y2) / 2
print(f'=== 整体模型评估结果 ===')
print(f'平均RMSE: {avg_rmse:.4f}')

# 将测试集指标写入TensorBoard
writer.add_text(
    tag='Test Set Metrics',
    text_string=f'''
    Y1（热负荷）- RMSE: {rmse_y1:.4f}, R²: {r2_y1:.4f}
    Y2（冷负荷）- RMSE: {rmse_y2:.4f}, R²: {r2_y2:.4f}
    平均RMSE: {avg_rmse:.4f}
    ''',
    global_step=epochs
)

# 关闭SummaryWriter释放资源
writer.close()


# 绘图
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 9))

# Y1（热负荷）：真实值 vs 预测值
ax1.scatter(y1_true, y1_pred, alpha=0.6, color='blue', label='预测值')
ax1.plot([y1_true.min(), y1_true.max()], [y1_true.min(), y1_true.max()],
         'r--', linewidth=2)
ax1.set_xlabel('真实热负荷（Y1）')
ax1.set_ylabel('预测热负荷（Y1）')
ax1.set_title(f'Y1 真实值 vs 预测值（RMSE={rmse_y1:.2f}, R方={r2_y1:.2f}）')
ax1.legend()
ax1.grid(alpha=0.3)

# Y2（冷负荷）：真实值 vs 预测值
ax2.scatter(y2_true, y2_pred, alpha=0.6, color='green', label='预测值')
ax2.plot([y2_true.min(), y2_true.max()], [y2_true.min(), y2_true.max()],
         'r--', linewidth=2)
ax2.set_xlabel('真实冷负荷（Y2）')
ax2.set_ylabel('预测冷负荷（Y2）')
ax2.set_title(f'Y2 真实值 vs 预测值（RMSE={rmse_y2:.2f}, R方={r2_y2:.2f}）')
ax2.legend()
ax2.grid(alpha=0.3)

# Y1 误差（真实值 - 预测值）
error_y1 = y1_true - y1_pred
ax3.hist(error_y1, bins=20, edgecolor='black', alpha=0.6, color='blue')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='误差=0')
ax3.set_xlabel('Y1 预测误差（真实值 - 预测值）')
ax3.set_ylabel('样本数量')
ax3.set_title(f'Y1 误差分布（均值={error_y1.mean():.2f}）')
ax3.legend()

# Y2 误差
error_y2 = y2_true - y2_pred
ax4.hist(error_y2, bins=20, edgecolor='black', alpha=0.6, color='green')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='误差=0')
ax4.set_xlabel('Y2 预测误差（真实值 - 预测值）')
ax4.set_ylabel('样本数量')
ax4.set_title(f'Y2 误差分布（均值={error_y2.mean():.2f}）')
ax4.legend()

plt.tight_layout()

# optimizer_name = 'SGD'
# lr = 0.01
# filename = f'../img/optimizer_{optimizer_name}_lr{lr}_epochs{epochs}.png'
# plt.savefig(filename, dpi=300, bbox_inches='tight')


plt.show()


'''
SGD 和 Adam 使用了不同的学习率，确实会影响比较的公平性。学习率是优化器最重要的超参数之一，
不同优化器对学习率的敏感度不同，因此直接用各自默认或任意设定的学习率对比，不一定能反映优化器本身的优劣。
tensorboard --logdir=logs

'''