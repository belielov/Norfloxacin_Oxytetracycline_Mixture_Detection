import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data(data_path):
    """ 加载光谱数据及对应标签，返回 Numpy 数组

    Args:
        data_path(str)：数据根目录路径（需包含以类别命名的子文件夹）

    Returns:
        tuple: (x, y) 组成的元组，其中：
            - x(np.ndarray): 光谱数据数组，形状为（样本数，特征数）
            - y(np.ndarray): 标签数组，形状为（样本数，）

    Raise:
        FileNotFoundError: 当指定的类别文件夹不存在时抛出异常
    """
    # 定义三个分类类别名称
    categories = ['Mixture', 'Norfloxacin', 'Oxytetracycline']

    # 初始化数据存储列表
    x = []  # 存储光谱数据特征（每个样本的第二列数据）
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
# project_dir = os.path.dirname(current_dir)
# data_path = os.path.join(project_dir, 'dataset', 'Excel')
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

# 构建输出路径
output_excel = os.path.join(project_dir, 'Results_Excel')
output_imgs = os.path.join(project_dir, 'imgs')

# 构建输出目录（如果不存在则自动创建）
# exist_ok=True：目录已存在时不抛出错误
os.makedirs(output_excel, exist_ok=True)
os.makedirs(output_imgs, exist_ok=True)

# # 测试：打印路径
# print(f'当前脚本所在目录路径：{current_dir}')
# print(f'项目根目录路径：{project_dir}')
# print(f'数据文件存储路径：{data_path}')
# print(f'输出excel文件存储路径：{output_excel}')
# print(f'输出图像文件存储路径：{output_imgs}')

# 加载数据
x, y = load_data(data_path)

# ------------------- 数据预处理 -------------------

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,  # 测试集占比20%
    stratify=y,  # 按原始标签比例分层抽样
    random_state=42  # 固定随机种子确保结果可复现
)

# 创建标准化器对象（Z-score标准化：均值为0，方差为1）
scaler = StandardScaler()

# 对训练集进行拟合转换：计算均值方差并应用标准化
x_train_scaled = scaler.fit_transform(x_train)

# 对测试集应用相同的标准化参数（使用训练集的均值方差）
x_test_scaled = scaler.transform(x_test)

# # 测试：打印预处理后的数据
# print(f'''
# x_train:{x_train}
# x_test:{x_test}
# y_train:{y_train}
# y_test:{y_test}
# x_train_scaled:{x_train_scaled}
# x_test_scled:{x_test_scaled}
# ''')

# ------------------- 线性判别分析(LDA)处理 -------------------

# 创建LDA降维器，指定降维到2个线性判别式
lda = LinearDiscriminantAnalysis(n_components=2)

# 在标准化后的训练数据上拟合LDA模型并进行转换
# LDA是监督学习方法，必须传入标签
x_train_lda = lda.fit_transform(x_train_scaled, y_train)

# 测试集必须使用与训练集相同的LDA参数
x_test_lda = lda.transform(x_test_scaled)

# 将训练集和测试集的LDA坐标垂直堆叠，创建统一的数据框
# np.vstack() 按行方向拼接数组（保持列数相同）
# 列名LD1、LD2表示第一个和第二个线性判别式
df_lda = pd.DataFrame(
    np.vstack([x_train_lda, x_test_lda]),
    columns=['LD1', 'LD2']
)

# 添加数据集来源标识列（前N行标记为Train，其余行标记为Test）
# 通过列表乘法快速生成对应长度的标签列表
df_lda['Dataset'] = ['Train']*len(x_train_lda) + ['Test']*len(x_test_lda)

# 将LDA坐标数据保存到excel文件，不保留行索引
df_lda.to_excel(os.path.join(output_excel, 'LDA_Coordinates.xlsx'), index=False)

# # 测试：打印LDA转换后的光谱数据
# print(f'经过LDA转换后的光谱数据：\n{df_lda}')

# ------------------- LDA可视化 -------------------

# 创建绘图画布
plt.figure(figsize=(10, 8))

# 定义可视化参数
colors = ['red', 'green', 'blue']
labels = ['Mixture', 'Norfloxacin', 'Oxytetracycline']

# # 测试：x_train_lda[y_train == i, 0]
# class_mask = (y_train == 0)
# print(f'y_train:\n{y_train}')
# print(f'类别0的布尔掩码：\n{class_mask}')
# print(f'属于类别0的样本在LD1轴上的坐标：\n{x_train_lda[class_mask, 0]}')

# 绘制训练集样本散点图（使用实心圆形标记）
for i in range(3):  # 遍历三个类别
    plt.scatter(
        x_train_lda[y_train == i, 0],  # 当前类别样本在LD1轴的坐标
        x_train_lda[y_train == i, 1],  # 当前类别样本在LD2轴的坐标
        color=colors[i],  # 设置颜色
        label=labels[i],  # 图例标签
        alpha=0.7,  # 透明度
        edgecolors='w',  # 点边缘颜色
        s=100  # 点大小
    )

# 绘制测试集样本散点图
for i in range(3):
    plt.scatter(
        x_test_lda[y_test == i, 0],
        x_test_lda[y_test == i, 1],
        color=colors[i],  # 保持与训练集相同的颜色
        marker='x',  # 使用x形标记区分测试集
        s=80,
        linewidths=1
    )

# 解释方差比：表示每个线性判别方向（LD）对分类能力的贡献比例，总和 ≤ 1。
plt.xlabel(f'LD1(解释方差比：{lda.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'LD2(解释方差比：{lda.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend()
plt.grid(True, alpha=0.3)  # 添加半透明网格线
plt.savefig(os.path.join(output_imgs, 'LDA_Visualization.png'))
plt.close()  # 关闭当前图形释放内存

# -------------------- SVM模型训练（基于LDA数据） --------------------

# 定义超参数网格，用于网格搜索（Grid Search）
# C：正则化参数，控制分类器的容错能力（值越大，容错越弱，可能过拟合）
# gamma：核函数系数（'scale'表示根据特征方差自动调整，'auto'为1/n_features）
param_grid = {
    'C': [0.1, 0.5, 1, 5, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# 初始化网格搜索对象
grid_search = GridSearchCV(
    SVC(),                # 使用支持向量分类器（SVC）
    param_grid,           # 超参数搜索空间
    cv=5,                 # 5折交叉验证
    scoring='accuracy',   # 以准确率作为评估指标
    n_jobs=-1             # 使用所有CPU核心并行加速
)

# 执行网格搜索：在LDA降维后的训练集上寻找最优参数组合
grid_search.fit(x_train_lda, y_train)

# 获取最优参数的SVM模型
best_svm = grid_search.best_estimator_

# 使用最优模型对LDA降维后的测试集进行预测
y_pred = best_svm.predict(x_test_lda)

# # 测试：打印测试标签和预测标签
# print(f'测试标签 y_test 为：\n{y_test}')
# print(f'预测标签 y_pred 为：\n{y_pred}')
# print(f'预测结果掩码为：\n{y_test == y_pred}')

# 保存最优模型到文件
dump(best_svm, 'best_svm_model.joblib')

# -------------------- 结果分析 --------------------

# 生成归一化混淆矩阵（按实际标签行归一化）
cm = confusion_matrix(y_test, y_pred, normalize='true')

# 将混淆矩阵转换为带类别标签的DataFrame
cm_df = pd.DataFrame(
    cm,
    index=labels,  # 行索引为真实类别名称
    columns=labels  # 列索引为预测类别名称
)
# 保存混淆矩阵到excel
cm_df.to_excel(os.path.join(output_excel, 'Confusion_Matrix.xlsx'))

# # 测试：打印混淆矩阵
# print(f'混淆矩阵为：\n{cm}')
# print(f'带类别标签的混淆矩阵：\n{cm_df}')

# 可视化混淆矩阵（使用蓝色渐变配色）
plt.figure(figsize=(12, 10))
ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap='Blues')  # 创建混淆矩阵热图
plt.title('SVM混淆矩阵（LDA特征）')
plt.savefig(os.path.join(output_imgs, 'SVM_Confusion_Matrix.png'), bbox_inches='tight')
plt.close()   # 关闭当前图像防止内存泄漏

# 生成分类报告（包含精确率、召回率、F1等指标）
report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)

# 将报告字典转换为DataFrame并保存（转置使指标列为行）
pd.DataFrame(report).transpose().to_excel(
    os.path.join(output_excel, 'SVM_Classification_Report.xlsx')
)

# # 测试：打印分类报告
# print(f'分类报告：\n{report}')

# ------------------- 交叉验证模型稳定性 -------------------

# 使用最优参数重新初始化模型
optimized_svm = SVC(**grid_search.best_params_)

# 执行5折交叉验证计算多个指标
# scoring参数指定需要计算的评估指标（使用宏平均）
# 宏平均计算方式：对每个类别单独计算指标后取算术平均。
# 宏平均特点：平等对待所有类别，适合类别分布相对平衡的场景。
cv_metrics = cross_validate(
    optimized_svm,
    x_train_lda,
    y_train,
    cv=5,
    scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
)

# # 测试：打印交叉验证指标
# print(f'交叉验证指标：\n{cv_metrics}')

# 将交叉验证结果转换为DataFrame（计算各指标均值）
metrics_df = pd.DataFrame({
    'Accuracy': cv_metrics['test_accuracy'].mean(),  # 平均准确率
    'Precision': cv_metrics['test_precision_macro'].mean(),  # 平均精确率
    'Recall': cv_metrics['test_recall_macro'].mean(),  # 平均召回率
    'F1': cv_metrics['test_f1_macro'].mean()  # 平均 f1 分数
}, index=[0])  # index设为单行方便查看

# 保存指标结果到Excel（不保留索引）
metrics_df.to_excel(os.path.join(output_excel, 'SVM_Performance_Metrics.xlsx'), index=False)

# -------------------- 关键信息输出 --------------------
print(f'''
=== 模型关键信息 ===
最佳SVM参数：{grid_search.best_params_}
测试集准确率：{best_svm.score(x_test_lda, y_test):.2%}
交叉验证指标：
  - 准确率：{metrics_df.Accuracy[0]:.2%}
  - 精确率：{metrics_df.Precision[0]:.2%}
  - 召回率：{metrics_df.Recall[0]:.2%}
  - F1分数：{metrics_df.F1[0]:.2%}
LDA解释方差比：{lda.explained_variance_ratio_}
''')
