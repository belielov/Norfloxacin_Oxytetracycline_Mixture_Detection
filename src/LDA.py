# 导入所需库
import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体（Windows系统）
plt.rcParams['axes.unicode_minus'] = False  # 关闭Unicode负号显示，强制使用ASCII的减号字符，解决负号'-'显示为方块的问题


def load_data(data_path):
    """
    加载数据集并生成特征矩阵和标签向量

    Args:
        data_path(str): 数据集根目录路径

    Returns:
        x(np.array):特征矩阵，形状为(n_samples, n_features)
        y(np.array):标签向量，形状为(n_samples)
    """
    categories = ['Mixture', 'Norfloxacin', 'Oxytetracycline']  # 定义三个类别名称
    x, y = [], []  # 初始化特征矩阵和标签列表

    # 遍历每个类别目录
    for label, category in enumerate(categories):
        category_dir = os.path.join(data_path, category)  # 构建类别完整路径

        # 检查目录是否存在
        if not os.path.exists(category_dir):
            raise FileNotFoundError(f'文件夹不存在：{category_dir}')

        # 遍历类别目录下的所有文件
        for file in os.listdir(category_dir):
            if file.endswith('.xlsx'):  # 仅处理Excel文件
                file_path = os.path.join(category_dir, file)  # 构建文件完整路径
                df = pd.read_excel(file_path, header=None)  # 读取excel文件（无表头）
                x.append(df[1].values)  # 提取第二列数据作为特征
                y.append(label)  # 添加标签（0，1，2对应三个类别）
    return np.array(x), np.array(y)  # 将列表转换为numpy数组并返回

# ---------------- 主程序 ----------------


# 设置项目路径
# dirname()：提取路径上级目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本目录
project_root = os.path.dirname(current_dir)  # 获取项目根目录
data_path = os.path.join(project_root, 'dataset', 'Excel')  # 组合数据集完整路径

# 加载数据
x, y = load_data(data_path)

# 数据标准化（LDA对特征尺度敏感）消除不同特征之间的量纲差异
scaler = StandardScaler()  # 创建标准化器
x_scaled = scaler.fit_transform(x)  # 拟合并转换数据（均值为0，方差为1）

# 使用LDA进行降维
lda = LinearDiscriminantAnalysis(n_components=2)  # 创建LDA模型，降维至2维
x_lda = lda.fit_transform(x_scaled, y)  # 监督学习需要传入标签

# 导出LDA降维后的LD1、LD2坐标
df_lda = pd.DataFrame(x_lda, columns=['LD1', 'LD2'])  # 创建包含坐标的DataFrame
output_path = os.path.join(project_root, 'Results_Excel', 'LDA_axis.xlsx')  # 设置输出路径为项目根目录
df_lda.to_excel(output_path, index=False)  # 保存Excel文件，不包含索引列

# ---------------- 可视化 ----------------
plt.figure(figsize=(10, 8))

# 定义可视化参数
colors = ['red', 'green', 'blue']  # 每个类别颜色
labels = ['Mixture', 'Norfloxacin', 'Oxytetracycline']  # 类别标签

# 绘制二维散点图
for i in range(3):
    plt.scatter(
        x_lda[y == i, 0],  # 第 i 类样本在LD1上的坐标
        x_lda[y == i, 1],  # 第 i 类样本在LD2上的坐标
        c=colors[i],  # 颜色
        label=labels[i],  # 标签
        alpha=0.7,  # 透明度
        edgecolors='w',  # 点边框白色
        s=100  # 点大小
    )

# 设置坐标轴标签（显示方差解释比例）
plt.xlabel(f'LD1(解释方差比：{lda.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'LD2(解释方差比：{lda.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('LDA聚类分析结果')
plt.legend(loc='upper right')  # 显示图例在右上角
plt.grid(True, alpha=0.3)  # 显示半透明网格


# ---------------- 添加决策边界 ----------------
# 使用降维后的数据重新训练LDA是为了绘制决策边界
lda_2d = LinearDiscriminantAnalysis()  # 创建新的LDA分类器
lda_2d.fit(x_lda, y)  # 在降维后的数据上训练

# 生成网格点用于绘制决策边界
x_min, x_max = x_lda[:, 0].min() - 1, x_lda[:, 0].max() + 1
y_min, y_max = x_lda[:, 1].min() - 1, x_lda[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),  # 生成x轴网格点
    np.arange(y_min, y_max, 0.02)  # 生成y轴网格点
)

# 预测网格点的类别
z = lda_2d.predict(np.c_[xx.ravel(), yy.ravel()])  # 展平网格并预测
z = z.reshape(xx.shape)  # 调整形状与网格一致

# 绘制决策边界区域
plt.contourf(xx, yy, z, alpha=0.1, cmap=plt.cm.Paired)  # 填充颜色区域
plt.show()

# ---------------- 模型评估 ----------------
print(f'类别先验概率：{lda.priors_}')  # 各类别在数据中的比例
print(f'解释方差比：{lda.explained_variance_ratio_}')  # 各主成分解释的方差比例
print(f'训练集分类准确率：{lda.score(x_scaled, y):.2%}')

