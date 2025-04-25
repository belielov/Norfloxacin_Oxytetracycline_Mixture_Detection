import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


def load_data(data_path):
    """ 加载光谱数据及对应标签，返回 Numpy 数组

    Args:
        data_path(str): 数据根目录路径（需包含以类别命名的子文件夹）

    Returns:
        tuple: (x, y) 组成的元组，其中：
            - x(np.ndarray): 光谱数据数组，形状为（样本数，特征数）
            - y(np.ndarray): 标签数组，形状为（样本数，）

    Raise：
        FileNotFoundError: 当指定的类别文件夹不存在时抛出异常
    """
    # 定义三个分类类别名称
    categories = ['Mixture', 'Norfloxacin', 'Oxytetracycline']

    # 初始化数据存储列表
    x = []  # 存储光谱特征数据（每个样本的第二列数据）
    y = []  # 存储对应的类别标签(0, 1, 2)

    # 遍历每个类别（使用enumerate同时获取标签索引和类别名）
    for label, category in enumerate(categories):
        # # 测试：打印 label, category
        # print(f'当前遍历的类别为：{label} {category}')

        # 构建当前类别的文件夹路径
        category_dir = os.path.join(data_path, category)

        # 检查文件夹是否存在
        if not os.path.exists(category_dir):
            raise FileNotFoundError(f'文件夹不存在：{category_dir}')

        # 遍历当前类别文件夹的所有文件
        for file in os.listdir(category_dir):
            # 仅处理.xlsx 文件
            if file.endswith('.xlsx'):
                # 构建完整文件路径
                file_path = os.path.join(category_dir, file)

                # # 测试：打印当前遍历的文件
                # print(f'当前遍历的文件为：{file_path}')

                # 读取excel文件（不包含表头）
                df = pd.read_excel(file_path, header=None)

                # 提取第二列数据
                x.append(df[1].values)  # df[1]表示第二列，.values转换为numpy数组

                # 添加对应的类别标签
                y.append(label)

    # 将列表转换为Numpy数组（x转换为二维数组，y转换为一维数组）
    return np.array(x), np.array(y)

# # 测试 load_data() 函数
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# data_path = os.path.join(project_root, 'dataset', 'Excel')
# data = load_data(data_path)
# print(f'数据路径为：{data_path}')
# print(f'加载的数据如下：\n{data}')


# 获取当前脚本的绝对路径并提取所在目录路径
# __file__ 表示当前执行脚本文件名，os.path.abspath()获取绝对路径
# os.path.dirname()提取父级目录路径（当前脚本所在目录）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录路径
project_dir = os.path.dirname(current_dir)

# 构建数据文件存储路径
data_path = os.path.join(project_dir, 'dataset', 'Excel')

# 构建输出路径：
# - Results_Excel/: 结果Excel文件输出目录
# - imgs/: 图像文件输出目录
output_excel = os.path.join(project_dir, 'Results_Excel')
output_imgs = os.path.join(project_dir, 'imgs')

# 创建输出目录（如果不存在则自动创建）
# exist_ok=True: 目录已存在时不抛出错误
os.makedirs(output_excel, exist_ok=True)
os.makedirs(output_imgs, exist_ok=True)

# # 测试：打印路径
# print(f'''
# 当前脚本文件所在目录路径：{current_dir}
# 项目根目录路径：{project_dir}
# 数据文件存储路径：{data_path}
# 输出excel文件存储路径：{output_excel}
# 输出图像文件存储路径：{output_imgs}
# ''')

# 加载数据
x, y = load_data(data_path)

# ------------------- 数据预处理 -------------------

# 创建标准化器对象（Z-score标准化：均值为零，方差归一）
scaler = StandardScaler()

# 进行拟合转换：计算均值方差并应用标准化
x_scaled = scaler.fit_transform(x)

# ------------------- PLS-DA（模拟OPLS-DA） -------------------

# 初始化PLS回归模型，设置提取2个潜变量（Latent Variables）
# n_components决定了降维后的维度，同时考虑自变量与因变量的协方差关系
pls = PLSRegression(n_components=2)

# 拟合PLS模型（训练过程）
# x_scaled: 标准化后的特征矩阵，y: 对应的目标变量/类别标签
# 模型会寻找最大化X与Y协方差的投影方向
pls.fit(x_scaled, y)

# 获取PLS的X空间得分矩阵（样本在潜变量空间的投影）
# scores的维度为(n_samples, n_components)，可用于可视化或后续分析
# 每个得分对应样本在潜变量方向上的坐标，类似于PCA中的主成分得分
scores = pls.x_scores_

# 将降维后的数据转换为DataFrame，并指定列名
df_opls = pd.DataFrame(scores, columns=['OPLS-DA1', 'OPLS-DA2'])
# 定义输出文件路径
output_path = os.path.join(output_excel, 'OPLS-DA_axis.xlsx')
# 将PCA结果保存到Excel文件（不包含索引）
df_opls.to_excel(output_path, index=False)

# ------------------- PLS-DA可视化 -------------------

# 创建8x6英寸的画布
plt.figure(figsize=(8, 6))
# 定义类别颜色和对应标签
colors = ['red', 'green', 'blue']
labels = ['Mixture', 'Norfloxacin', 'Oxytetracycline']

# 循环绘制三个类别的散点图
for i in range(3):
    plt.scatter(
        scores[y == i, 0],
        scores[y == i, 1],
        c=colors[i],
        label=labels[i]
    )

# 设置坐标轴标签
plt.xlabel('Component 1 (OPLS-DA1)')
plt.ylabel('Component 2 (OPLS-DA2)')
plt.legend()
plt.title('OPLS-DA Clustering Results')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_imgs, 'OPLS-DA.png'))
plt.close()  # 关闭当前图形释放内存

# ------------------- PLS-DA评价指标 -------------------

# 导入评估指标计算模块
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 预测样本类别（注意：PLS回归输出连续值需要转换为离散标签）
y_pred = pls.predict(x_scaled)          # 获取连续预测值
y_pred = np.round(y_pred).astype(int)   # 四舍五入转换为整数标签
y_pred = np.clip(y_pred, 0, 2)          # 限制标签在0-2范围内

# 计算分类指标
accuracy = accuracy_score(y, y_pred)    # 总体准确率
conf_matrix = confusion_matrix(y, y_pred)  # 混淆矩阵
class_report = classification_report(y, y_pred)  # 分类报告（含精确率、召回率等）

# 计算X方差解释率（R²X）
# 原理：计算潜变量对原始X方差的解释比例
x_scores = pls.x_scores_  # 获取样本在潜变量空间的坐标
x_total_var = np.var(x_scaled, ddof=1, axis=0).sum()  # 计算X总方差（无偏估计）
x_explained_var = np.var(x_scores, axis=0, ddof=1)    # 各潜变量解释的方差
r2_x = x_explained_var / x_total_var                  # 计算方差解释比例

# 计算Y决定系数（R²Y）
# 注意：此处使用回归模型的score方法，反映模型对Y的拟合程度
r2y = pls.score(x_scaled, y)  # 返回决定系数R²

# 打印评价指标
print('\n======= PLS-DA 评价指标 =======')
print(f'分类准确率: {accuracy:.3f}')
print(f'X方差解释率(R²X): {r2_x}')
print(f'累计X方差解释率: {sum(r2_x):.3f}')
print(f'Y决定系数(R²Y): {r2y:.3f}\n')
print('混淆矩阵:')
print(conf_matrix)
print('\n分类报告:')
print(class_report)
