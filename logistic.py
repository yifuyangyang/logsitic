import numpy as np
from sklearn.linear_model import LogisticRegression


#filename=r"C:\Users\E507\Desktop\logsitic\horseColicTraining.txt"
#=====================
# 1. 数据读取函数
#=====================
def load_dataset(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]   # 特征
    y = data[:, -1]    # 标签
    return X, y

#=====================
# 2. 缺失值处理函数
#   （缺失值替换为该列均值）
#=====================
def replace_nan_with_mean(X):
    for i in range(X.shape[1]):
        col = X[:, i]
        # 选择非0的数作为有效特征
        valid = col[col != 0]
        if len(valid) > 0:
            mean_val = np.mean(valid)
            col[col == 0] = mean_val
            X[:, i] = col
    return X

#=====================
# 3. 主流程
# 读取训练集
X_train, y_train = load_dataset(r"C:\Users\E507\Desktop\logsitic\horseColicTraining.txt")
# 读取测试集
X_test, y_test = load_dataset(r"C:\Users\E507\Desktop\logsitic\horseColicTest.txt")

# 处理训练集缺失值
X_train = replace_nan_with_mean(X_train)
# 处理测试集缺失值
X_test = replace_nan_with_mean(X_test)

#=====================
# 4. 构建并训练逻辑回归模型
#=====================
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

#=====================
# 5. 测试集预测
#=====================
y_pred = model.predict(X_test)

#=====================
# 6. 计算准确率
#=====================
accuracy = np.mean(y_pred == y_test)
print(f"测试集准确率: {accuracy:.4f}")


