
import numpy as np
from sklearn.linear_model import LogisticRegression

# 数据集路径（根据你的实际路径修改，此处保持你提供的路径）
train_filename = r"C:\Users\E507\Documents\GitHub\logsitic\horseColicTraining.txt"
test_filename = r"C:\Users\E507\Documents\GitHub\logsitic\horseColicTest.txt"

# =====================
# 1. 数据读取函数
# =====================
def load_dataset(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]  # 特征（所有列除了最后一列）
    y = data[:, -1]   # 标签（最后一列）
    return X, y

# =====================
# 2. 缺失值处理函数
# （缺失值替换为该列均值，此处默认0为缺失值标记）
# =====================
def replace_nan_with_mean(X):
    for i in range(X.shape[1]):
        col = X[:, i]
        # 选择非0的数作为有效特征（假设0代表缺失值）
        valid = col[col != 0]
        if len(valid) > 0:
            mean_val = np.mean(valid)
            col[col == 0] = mean_val  # 用均值替换缺失值
            X[:, i] = col
    return X

# =====================
# 3. 主流程
# =====================
# 读取训练集并处理缺失值
X_train, y_train = load_dataset(train_filename)
X_train = replace_nan_with_mean(X_train)

# 读取测试集并处理缺失值
X_test, y_test = load_dataset(test_filename)
X_test = replace_nan_with_mean(X_test)

# =====================
# 4. 构建并训练逻辑回归模型
# =====================
# 初始化逻辑回归模型（ solver='liblinear' 适配小数据集，避免警告）
model = LogisticRegression(solver='liblinear', max_iter=100)
model.fit(X_train, y_train)  # 训练模型

# =====================
# 5. 测试集预测
# =====================
y_pred = model.predict(X_test)  # 预测测试集标签

# =====================
# 6. 计算准确率
# =====================
accuracy = np.mean(y_pred == y_test)  # 计算预测正确的比例
print(f"测试集准确率: {accuracy:.4f}")  # 输出准确率（保留4位小数）