import os.path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


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

# 获取当前脚本文件的绝对路径并提取所在目录路径
# __file__ 表示当前执行脚本的文件名，os.path.abspath() 获取绝对路径
# os.path.dirname() 提取父级目录路径（当前脚本所在目录）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录路径（当前目录的父级目录）
# 假设项目结构为：
# project_root/
# ├── src/         <- 当前脚本所在目录
# ├── dataset/
# ├── Results_Excel/
# └── imgs/
project_dir = os.path.dirname(current_dir)

# 构建数据文件存储路径：项目根目录/dataset/Excel
# 用于存放原始数据Excel文件
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
# print(f'当前脚本文件所在目录路径：{current_dir}')
# print(f'项目根目录路径：{project_dir}')
# print(f'数据文件存储路径：{data_path}')
# print(f'输出excel文件存储路径：{output_excel}')
# print(f'输出图像文件存储路径：{output_imgs}')

# 加载数据
x, y = load_data(data_path)

# ------------------- 数据预处理 -------------------

# 划分训练集和测试集（保持类别分布平衡）
# 参数说明:
# - test_size=0.2 : 测试集占比20%
# - stratify=y     : 按原始标签比例分层抽样（保持类别分布）
# - random_state=42: 固定随机种子确保结果可复现
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 创建标准化器对象（Z-score标准化：均值为零，方差归一）
scaler = StandardScaler()

# 对训练集进行拟合转换：计算均值方差并应用标准化
# 注意：只对训练集使用fit_transform，避免数据泄露
x_train_scaled = scaler.fit_transform(x_train)

# 对测试集应用相同的标准化参数（使用训练集的均值和方差）
# 重要：测试集不应单独计算标准化参数，必须使用训练集的参数
x_test_scaled = scaler.transform(x_test)

# # 测试：打印预处理后的数据
# print(f'x_train:{x_train}')
# print()
# print(f'x_test:{x_test}')
# print()
# print(f'y_train:{y_train}')
# print()
# print(f'y_test:{y_test}')
# print()
# print(f'x_train_scaled:{x_train_scaled}')
# print()
# print(f'x_test_scaled:{x_test_scaled}')

# ------------------- 线性判别分析(LDA)处理 -------------------

# 创建LDA降维器，指定降维到2个线性判别式（适合二维可视化）
# n_components=2 表示将数据投影到前两个判别向量构成的空间
lda = LinearDiscriminantAnalysis(n_components=2)

# 在标准化后的训练数据上拟合LDA模型并进行转换（同时使用特征和标签）
# 注意：LDA是监督学习方法，需要传入标签y_train
x_train_lda = lda.fit_transform(x_train_scaled, y_train)

# 对标准化后的测试数据应用训练好的LDA转换（仅使用transform）
# 重要：测试集必须使用与训练集相同的LDA参数
x_test_lda = lda.transform(x_test_scaled)

# 将训练集和测试集的LDA坐标垂直堆叠，创建统一的数据框
# np.vstack 按行方向拼接数组（保持列数相同）
# 列名LD1/LD2表示第一个和第二个线性判别式
df_lda = pd.DataFrame(
    np.vstack([x_train_lda, x_test_lda]),
    columns=['LD1', 'LD2']
)

# 添加数据集来源标识列（前N行标记为Train，剩余行标记为Test）
# 通过列表乘法快速生成对应长度的标签列表
df_lda['Dataset'] = ['Train']*len(x_train_lda) + ['Test']*len(x_test_lda)

# 将LDA坐标数据保存到Excel文件（不保留行索引）
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
# print(f'类别0的布尔掩码：\n{class_mask}')
# print(f'属于类别0的样本在LD1轴的坐标：\n{x_train_lda[class_mask, 0]}')

# 绘制训练集样本散点图（使用实心圆形标记）
for i in range(3):  # 遍历三个类别
    plt.scatter(
        x_train_lda[y_train == i, 0],  # 当前类别样本在LD1轴的坐标
        x_train_lda[y_train == i, 1],  # 当前类别样本在LD2轴的坐标
        c=colors[i],  # 设置颜色
        label=labels[i],  # 图例标签
        alpha=0.7,  # 透明度（0-1之间）
        edgecolors='w',  # 点边缘颜色（白色描边）
        s=100  # 点大小
    )

# 绘制测试集样本散点图（使用x形标记）
for i in range(3):
    plt.scatter(
        x_test_lda[y_test == i, 0],  # 测试集在LD1轴的坐标
        x_test_lda[y_test == i, 1],  # 测试集在LD2轴的坐标
        c=colors[i],  # 保持与训练集相同的颜色
        marker='x',  # 使用x形标记区分测试集
        s=80,
        linewidths=1  # 标记线宽
    )

# 解释方差比：表示每个线性判别方向（LD）对分类能力的贡献比例，总和 ≤ 1。
plt.xlabel(f'LD1(解释方差比：{lda.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'LD2(解释方差比：{lda.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('LDA降维结果（圆形：训练集，叉号：测试集）')
plt.legend()
plt.grid(True, alpha=0.3)  # 添加半透明网格线
plt.savefig(os.path.join(output_imgs, 'LDA_Visualization.png'))
plt.close()  # 关闭当前图形释放内存

# ------------------- XGBoost模型训练 -------------------

# 定义超参数搜索网格（关键参数组合空间）
# 注意：参数范围设置需平衡计算成本与搜索效果
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 学习率（步长收缩，防止过拟合）
    'n_estimators': [100, 200, 300],  # 基学习器（决策树）数量
    'max_depth': [3, 4, 5, 6],  # 单棵树最大深度（控制复杂度）
    'min_child_weight': [1, 3, 5],  # 叶子节点最小样本权重和（控制分裂）
    'subsample': [0.8, 1.0],  # 样本子采样比例（防止过拟合）
    'colsample_bytree': [0.8, 1.0]  # 特征子采样比例（每棵树使用的特征比例）
}

# 创建网格搜索对象（5折交叉验证）
grid_search = GridSearchCV(
    estimator=XGBClassifier(
        objective='multi:softmax',  # 多分类目标函数
        num_class=3,  # 类别数量
        random_state=42,  # 固定随机种子保证可复现
        eval_metric='mlogloss'  # 多分类对数损失评估指标
    ),
    param_grid=param_grid,  # 传入参数搜索空间
    cv=5,  # 5折交叉验证
    scoring='f1_macro',  # 评估指标（宏平均F1分数，适用于类别不平衡）
    n_jobs=-1,  # 使用所有CPU核心并行计算
    verbose=1  # 输出详细进度（1=显示进度条）
)

# 执行网格搜索（在LDA降维后的训练数据上进行）
grid_search.fit(x_train_lda, y_train)

# 获取最优模型（自动从所有参数组合中选择验证得分最高的）
best_xgb = grid_search.best_estimator_

# 使用最优模型进行测试集预测（使用LDA处理后的测试数据）
y_pred = best_xgb.predict(x_test_lda)

# # 测试：打印测试标签和预测标签
# print(f'测试标签 y_test 为：\n{y_test}')
# print(f'预测标签 y_pred 为：\n{y_pred}')
# print(f'预测结果掩码为：\n{y_test == y_pred}')

# 保存最佳模型
joblib.dump(best_xgb, os.path.join(current_dir, 'best_xgb_model.pkl'))

# ------------------- 结果分析 -------------------

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
plt.title('XGBoost混淆矩阵（LDA特征）')
plt.savefig(os.path.join(output_imgs, 'XGBoost_Confusion_Matrix.png'), bbox_inches='tight')  # 保存图像时自动调整边距
plt.close()  # 关闭当前图像防止内存泄漏

# 生成分类报告（包含精确率、召回率、F1等指标）
report = classification_report(
    y_test,
    y_pred,
    target_names=labels,  # 使用类别名称替代数字标签
    output_dict=True
)
# 将报告字典转换为DataFrame并保存（转置使指标列为行）
pd.DataFrame(report).transpose().to_excel(
    os.path.join(output_excel, 'XGBoost_Classification_Report.xlsx')
)

# # 测试：打印分类报告
# print(f'分类报告：\n{report}')

# ------------------- 交叉验证模型稳定性 -------------------

# 使用最优参数重新初始化模型
optimized_xgb = XGBClassifier(**grid_search.best_params_)

# 执行5折交叉验证计算多个指标
# scoring参数指定需要计算的评估指标（使用宏平均）
# 宏平均计算方式：对每个类别单独计算指标后取算术平均。
# 宏平均特点：平等对待所有类别，适合类别分布相对平衡的场景。
cv_metrics = cross_validate(
    optimized_xgb, x_train_lda, y_train, cv=5,
    scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
)

# # 测试：打印交叉验证指标
# print(f'交叉验证指标：\n{cv_metrics}')

# 将交叉验证结果转换为DataFrame（计算各指标均值）
metrics_df = pd.DataFrame({
    'Accuracy': cv_metrics['test_accuracy'].mean(),  # 平均准确率
    'Precision': cv_metrics['test_precision_macro'].mean(),  # 平均精确率
    'Recall': cv_metrics['test_recall_macro'].mean(),  # 平均召回率
    'F1_Score': cv_metrics['test_f1_macro'].mean()  # 平均 f1 分数
}, index=[0])  # index设为单行方便查看

# 保存指标结果到Excel（不保留索引）
metrics_df.to_excel(os.path.join(output_excel, 'XGBoost_Performance_Metrics.xlsx'), index=False)

# ------------------- 关键信息输出 -------------------

# # 测试：打印metrics_df.Accuracy
# print(metrics_df.Accuracy)

print(f'''
=== 模型关键信息 ===
最佳XGBoost参数：{grid_search.best_params_}
测试集准确率：{best_xgb.score(x_test_lda, y_test):.2%}
交叉验证指标：
    - 准确率：{metrics_df.Accuracy[0]:.2%}
    - 精确率：{metrics_df.Precision[0]:.2%}
    - 召回率：{metrics_df.Recall[0]:.2%}
    - F1分数：{metrics_df.F1_Score[0]:.2%}
LDA解释方差比：{lda.explained_variance_ratio_}
''')
