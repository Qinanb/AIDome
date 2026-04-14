#设置随机种子保证结果的可复现性
import numpy as np
#导入python随机数
import random
import os
import torch
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
    print('CUDA is useful!!')
else:
    print('CUDA is not useful!!')
#设置随机数种子
setup_seed(0)

#定义数据集的加载于处理
#定义训练数据的处理步骤
from torchvision import transforms
train_transforms=transforms.Compose([
    transforms.Resize((224,224)),#转成尺寸为224x224
    transforms.ToTensor(),#转成PIL image 到 tensor
    transforms.Normalize(#归一化常用的图像的均值和标准差
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225])
])
valid_transforms=transforms.Compose([
    transforms.Resize((224,224)),#转成尺寸为224x224
    transforms.ToTensor(),#转成PIL image 到 tensor
    transforms.Normalize(#归一化常用的图像的均值和标准差
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225])
])

#读取数据
from torchvision import datasets
# ImageFolder 的root属性为数据集的根目录，子文件夹名作为类别名。每个子文件夹内包含属于该类的图像
# ImageFolder 的transform属性为应用于图像的变换，用于图像的处理
train_dataset =datasets.ImageFolder(root='../dataset/train',transform=train_transforms)
valid_dataset = datasets.ImageFolder(root='../dataset/val', transform=valid_transforms)
#做成dataloader,方便后续模型的训练
from torch.utils.data import DataLoader
train_dataloader=DataLoader(dataset=train_dataset,batch_size=128,shuffle=True)
valid_dataloader=DataLoader(dataset=valid_dataset,batch_size=128,shuffle=True)
#从训练集中抽几张图片进行显示(可选)
# import matplotlib.pyplot as plt
# for batch_i,(imgs,labs) in enumerate(train_dataloader):
#     #展示一个批次的四张图片
#     for num in range(4):
#         #plt.subplot创建一个子图并激活它（索引从1开始）
#         plt.subplot(2,2,num+1)
#         plt.imshow(imgs[num][0],cmap='gray')
#         plt.title(f'Ground Truth: {labs[num]}')
#         plt.xticks([])
#         plt.yticks([])
#     plt.show()
#     break

#实例化模型
from torchvision.models import resnet18
#判断是否有gpu如果有gpu使用gpu没有使用cpu
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
#weights=None 表示构建一个ResNet18模型，但不加载任何预训练权重，也就是使用随机初始化的参数
model=resnet18(weights=None)
static_dict = torch.load(r'D:\pydome\AI25-12\03_Deep_Learning\01_CNN_Progect\01_基于ResNet的汽车分类\model\resnet18-5c106cde.pth', weights_only=True)
#使用模型加载状态字典
model.load_state_dict(static_dict)#加载权重
#冻结模型的所有参数，通常用于迁移学习中
for param in model.parameters():#返回模型中所有可训练参数(权重和偏置 )的迭代器
    param.requires_grad=False#表示该参数的不参与梯度计算，训练时权重保持不变

import torch.nn as nn
fc_input=model.fc.in_features
model.fc=nn.Sequential(
    nn.Linear(fc_input,256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256,10),
    nn.LogSoftmax(dim=1)
)
#将模型放到设备上(cpu或者gpu上)
model.to(device)
# print(model)
#定义损失函数和优化器
criterion=nn.CrossEntropyLoss()#交叉熵损失函数
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)#优化器
#定义模型保存路径
save_path='../model/best.pth'
save_dir=os.path.dirname(save_path) #提取保存路径中的目录部分 save_dir
#判断目录是否存在，不存在则创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 训练模型
num_epoch=10
for epoch in range(num_epoch):
    model.train()
    total_loss = 0
    for i,(images,labels) in enumerate(train_dataloader):
        #数据放到cpu或者gpu上
        images=images.to(device)
        labels=labels.to(device)
        #清楚历史梯度
        optimizer.zero_grad()
        #前向计算
        outputs=model(images)
        #计算损失
        loss=criterion(outputs,labels)
        #反向传播:计算参数梯度
        loss.backward()
        #参数更新：根据梯度调整参数
        optimizer.step()
        total_loss+=loss.item()
        print(f"Epoch[{epoch+1}/{num_epoch}] Batch [{i+1}/{len(train_dataloader)}] Loss {loss.item():.4f}")
    avg_loss=total_loss/(len(train_dataloader))
    if(epoch+1)%1==0:
        print(f"Epoch[{epoch+1}/{num_epoch}] Loss {avg_loss:.4f}")
#保持模型
torch.save(model.state_dict(),save_path)
#模型评估
model.eval()
correct=0
total=0
predicted_labels=[]
true_labels=[]
with torch.no_grad():
    for images,labels in valid_dataloader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        # 忽略第一个返回值（最大值），用_接收
        #predicted,每个元素是对应样本的预测类别索引
        _,predicted=torch.max(outputs.data,1)#沿着第一维度获得最大值(_)及最大值索引(predicted)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())#将列表的元素逐个添加到当前列表
        true_labels.extend(labels.cpu().numpy())
print(f'Accuracy of the model on test images: {100 * correct / total:.2f}%')
#可视化混淆矩阵
from sklearn.metrics import confusion_matrix
conf_matrix=confusion_matrix(true_labels,predicted_labels)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
