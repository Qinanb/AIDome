import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

# 加载数据集
from sklearn.datasets import load_breast_cancer

# 加载乳腺癌数据集，并将其转换为pandas DataFrame格式，以便进行数据处理和分析。数据集中包含了乳腺癌的特征和对应的类别标签（良性或恶性）。
# 我们将特征存储在DataFrame的列中，并将类别标签存储在一个新的列中，命名为'Class'。
data=load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Class'] = data.target


# 查看数据是否读取成功
print(df.head())

# 去掉数据中的异常值
df=df.replace('?', np.nan)  # 将'?'替换为NaN
df=df.dropna(axis=0)  # 删除包含NaN的行

# 特征集，排除目标变量
# 在这个数据集中，目标变量是'Class'，它表示乳腺癌的类型（良性或恶性）。
# 我们需要将'Class'列从特征集中分离出来，以便进行模型训练和评估。
# 因此，我们可以使用pandas的drop方法来删除'Class'列，并将剩余的列作为特征集X。同时，我们将'Class'列作为标签y。
x=df.drop(['Class'], axis=1) 
y=df['Class'] 

# 在乳腺癌这个数据集里，特征本来就是数值型连续变量，不是类别型变量，所以不需要做独热编码
# # 处理类别特征：使用独热编码，将类别特征转换为数值特征
# x_encoded=pd.get_dummies(x)
# print(x_encoded.keys())

# 划分数据集和测试集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# 使用标准化进行特征缩放（本例中都的特征都是离散型，实际上可以不进行归一化）
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
# 将标签进行编码，将类别标签转换为数值标签
label_encoder = LabelEncoder()
y_train_numeric = label_encoder.fit_transform(y_train)
y_test_numeric = label_encoder.transform(y_test)
# 将数据转换为PyTorch张量
x_train_tensor=torch.tensor(x_train_scaled, dtype=torch.float32)
y_train_tensor=torch.tensor(y_train_numeric, dtype=torch.float32).unsqueeze(1)  # 将标签转换为列向量
x_test_tensor=torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_tensor=torch.tensor(y_test_numeric, dtype=torch.float32).unsqueeze(1)  # 将标签转换为列向量

# 定义模型
# 定义模型
class SigmoidRegression(torch.nn.Module):
    def __init__(self,input_size):
        super(SigmoidRegression,self).__init__()
        # 定义线性层，输入特征数为input_size，输出类别数为1
        self.fc=torch.nn.Linear(input_size,1)
    def forward(self,x):
        # 前向传播，输出类别概率
        x=self.fc(x)
        return torch.sigmoid(x)
    
# 模型初始化 
model=SigmoidRegression(x_train_tensor.shape[1])

# 定义损失函数和优化器
# 使用二元交叉熵损失函数（BCELoss）来衡量模型的预测与实际标签之间的差距。
# 优化器选择Adam，它是一种常用的优化算法，可以自适应调整学习率以加速训练过程。
cri=torch.nn.BCELoss()
lr=0.001
optimizer=torch.optim.Adam(model.parameters(),lr=lr)

fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(12,4))
# 获取迭代次数和损失
epoch_list=[]
loss_list=[]
f1_score_list=[]
f1_epoch_list=[]
# 将测试标签转换为NumPy数组，以便计算F1分数
y_test_np = y_test_tensor.cpu().numpy().ravel()
# 训练模型
epochs=2000
for epoch in range(1,epochs+1):
    model.train()
    # 前向传播
    outputs=model(x_train_tensor)
    loss=cri(outputs,y_train_tensor)  # 将标签转换为浮点类型并去掉多余的维度
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 记录损失和迭代次数
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    
    if epoch % 100 == 0 or epoch == 1:
        

        # 测试模式
        model.eval()
        with torch.no_grad():
            # 在测试集上进行预测，并计算F1分数
            test_output = model(x_test_tensor)
            y_pred = (test_output > 0.5).float().cpu().numpy().ravel()
            f1 = f1_score(y_test_np, y_pred)
            f1_epoch_list.append(epoch)
            f1_score_list.append(f1)
            # 计算ROC曲线和AUC值
            test_score = test_output.cpu().numpy()
            fpr, tpr, threshold = roc_curve(y_test_np, test_score)
            roc_auc = auc(fpr, tpr)
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}, F1 Score: {f1:.4f}, AUC: {roc_auc:.4f}')

        # 绘制损失曲线
        ax1.clear()
        ax1.plot(epoch_list, loss_list, color='blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curve')

        # 绘制 F1 曲线
        ax2.clear()
        ax2.plot(f1_epoch_list, f1_score_list, color='red', label='F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Test F1 Score During Training')
        ax2.legend()
        ax2.set_ylim(0, 1)

        # 绘制当前迭代下的ROC曲线
        ax3.clear()
        ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'Epoch {epoch}, AUC = {roc_auc:.2f}')
        ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve During Training')
        ax3.legend(loc='lower right')
       

        plt.pause(0.01)  # 暂停以更新图形
plt.show()
