import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path='./dataset/ev_range_dataset.xlsx'
# path=r'AI25-12\Machine Learning\Day2\dataset\ev_range_dataset.xlsx'
data=pd.read_excel(path)
# print(data.head())
data=data.to_numpy()
x_data=data[:,0]
y_data=data[:,1]

# print(max(x_data)) # 500
# print(max(y_data)) # 76.82

# 损失函数
def loss(w,b,x_data,y_data):
    y_pre=w*x_data+b
    e_bar=np.mean((y_data-y_pre)**2)
    return e_bar

# 参数初始化
w=0.3
b=2

w_old=w # 第一次的w旧使用的初始参数
lr=0.000001 # 学习率

# 创建窗口子图
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
# 子图1的w轴范围
w_values=np.linspace(-0.5,0.5,200)
# 子图2的x轴范围
ax2_x_axis_min=0
ax2_x_axis_max=520

# 计算采样点的损失值
loss_values=[]
for w in w_values:
    ls=loss(w,b,x_data,y_data)
    loss_values.append(ls)

# 设置迭代器次数
iterations=50


def init_static_area():
    # 清空
    ax1.cla()
    ax2.cla()

    # 设置两个子图的坐标轴范围
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-500, 3000)
    ax1.set_xlabel('w')
    ax1.set_ylabel('loss')
    ax1.set_title('Loss Function')
    # 绘制损失函数曲线
    ax1.plot(w_values, loss_values, color='g', linewidth=2)

    ax2.set_xlim(ax2_x_axis_min, ax2_x_axis_max)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Data Scatter And Fitting Line')  # 数据散点和拟合线
    # 绘制数据散点图
    ax2.scatter(x_data, y_data, color='b')

# 循环迭代进行反向传播并绘制图像
for i in range(iterations):
    init_static_area() # 重置静态区域

    e_bar=loss(w_old,b,x_data,y_data) # 计算当前参数的损失值
    # 绘制左图红点
    ax1.plot(w_old,e_bar,color='r',marker='o',markersize=8)
    y_pred = w_old* x_data + b
    de_dw = -2 * np.mean(x_data * (y_data - y_pred))
    # de_dw=2*(np.mean((x_data**2)*w_old)-np.mean(x_data*y_data)) # 计算w的斜率
    intercept=e_bar-de_dw*w_old # 计算截距
    w_new=w_old-lr*de_dw # 计算w的新值
    w_old=w_new # 更新w旧的值
    print(f'w{w_new}')
    # 绘制切线
    line=de_dw*w_values+intercept
    ax1.plot(w_values,line,color='r',linestyle='--',linewidth=2)

    # 绘制右图红线
    y_lower=w_new*ax2_x_axis_min+b
    y_upper=w_new*ax2_x_axis_max+b
    ax2.plot(
        [ax2_x_axis_min,ax2_x_axis_max],
        [y_lower,y_upper],
        color='r',
        linewidth=2
    )
    plt.pause(0.1)
print(f'最佳线性关系方程：y={w_new:.3f}*x+2')
plt.show()
