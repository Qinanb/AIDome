from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# 设置随机数种子
seed=42
# 生成模拟数据
# make_blobs函数用于生成聚类数据集，其中n_samples参数指定了样本数量，
# centers参数指定了聚类中心的数量，random_state参数设置随机种子以确保结果的可重复性。
data=make_blobs(n_samples=400, centers=4, random_state=seed)
print(data)

# 设置聚类簇数
# k参数指定了KMeans算法中要形成的聚类簇的数量。在这个例子中，我们将k设置为4，表示我们希望将数据分成4个不同的簇。
k=4
# 创建KMeans模型

# max_iter参数指定了算法的最大迭代次数，random_state参数设置随机种子以确保结果的可重复性。
kmeans=KMeans(n_clusters=k,max_iter=30,random_state=seed)

# 训练模型
# fit方法用于训练KMeans模型，它会根据输入的数据进行聚类分析，并找到每个簇的中心点（centroids）。
# 在这个例子中，我们将数据集中的特征（data[0]）传递给fit方法，以便模型能够学习数据的结构并进行聚类。
kmeans.fit(data[0])

#获取簇心
# cluster_centers_属性返回一个数组，其中包含了每个簇的中心点坐标。
# 在这个例子中，我们将簇心存储在变量centeroids中，以便后续使用。
centeroids=kmeans.cluster_centers_  
pre_kmean=kmeans.predict(data[0])
print(pre_kmean)


# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(data[0][:, 0], data[0][:, 1], cmap='viridis', marker='o', edgecolor='k')
ax1.set_title('Original Data with True Labels')
ax2.scatter(data[0][:, 0], data[0][:, 1], c=pre_kmean, cmap='viridis', marker='o', edgecolor='k')
ax2.scatter(centeroids[:, 0], centeroids[:, 1], c='red', marker='*', s=100, label='Centroids')
ax2.set_title('K-Means Clustering Results')
ax2.legend()
plt.show()  