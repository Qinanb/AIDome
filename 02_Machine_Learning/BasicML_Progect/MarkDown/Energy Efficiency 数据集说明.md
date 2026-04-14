# Energy Efficiency 数据集说明

## 数据集基本信息

- **发布时间**：2012-11-29
- **数据集 ID**：242（UCI 机器学习仓库）
- **DOI**：10.24432/C51307
- **创建者**：Athanasios Tsanas、Angeliki Xifara
- **核心关联论文**：Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools（发表于*Energy and Buildings*, vol. 49, 2012）
- **核心用途**：通过建筑参数预测建筑热负荷、冷负荷需求，分析建筑能源效率，适用于回归 / 分类机器学习任务

## 数据集核心特征

|    特征项    |                 具体说明                 |
| :----------: | :--------------------------------------: |
|   数据类型   |          多变量（Multivariate）          |
|   所属领域   |                计算机科学                |
|   关联任务   |  回归（核心）、多分类（目标变量取整后）  |
|   特征类型   | 整数（Integer）、实数（Real/Continuous） |
|   样本总量   |                   768                    |
|   特征数量   |                   8 个                   |
| 目标变量数量 |              2 个（连续型）              |
|  缺失值情况  |                 无缺失值                 |

## 数据集背景

本数据集基于 Ecotect 软件模拟实现，共模拟 12 种不同建筑外形，建筑在玻璃面积、玻璃面积分布、建筑朝向等核心参数上存在差异；基于上述特征模拟多种参数设置，最终生成 768 个建筑样本数据，用于预测建筑能源负荷相关的两个连续型指标。

若将目标变量四舍五入至最近整数，该数据集也可适配多分类机器学习任务。

## 变量详细信息

| 变量名 | 角色 | 数据类型 |                   描述                    | 单位 | 缺失值 |
| :----: | :--: | :------: | :---------------------------------------: | :--: | :----: |
|   X1   | 特征 |  连续型  |    相对紧凑度（Relative Compactness）     |  -   |   无   |
|   X2   | 特征 |  连续型  |          表面积（Surface Area）           |  -   |   无   |
|   X3   | 特征 |  连续型  |           墙体面积（Wall Area）           |  -   |   无   |
|   X4   | 特征 |  连续型  |           屋顶面积（Roof Area）           |  -   |   无   |
|   X5   | 特征 |  连续型  |         总高度（Overall Height）          |  -   |   无   |
|   X6   | 特征 |  整数型  |            朝向（Orientation）            |  -   |   无   |
|   X7   | 特征 |  连续型  |         玻璃面积（Glazing Area）          |  -   |   无   |
|   X8   | 特征 |  整数型  | 玻璃面积分布（Glazing Area Distribution） |  -   |   无   |
|   Y1   | 目标 |  连续型  |          热负荷（Heating Load）           |  -   |   无   |
|   Y2   | 目标 |  连续型  |          冷负荷（Cooling Load）           |  -   |   无   |

## 数据集文件

|      文件名       | 文件大小 |
| :---------------: | :------: |
| ENB2012_data.xlsx | 74.4 KB  |

## 数据集引用格式（APA）

Tsanas, A. & Xifara, A. (2012). Energy Efficiency [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51307.

## 引用该数据集的相关论文

1. Yazici, M., Basurra, S., & Gaber, M. (2018). Edge Machine Learning: Enabling Smart Internet of Things Applications. *Big Data and Cognitive Computing*.
2. Zegklitz, J., & Posík, P. (2017). Symbolic Regression Algorithms with Built-in Linear Regression. *ArXiv*.