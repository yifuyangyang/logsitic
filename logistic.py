import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 


fileText=r"C:\Users\E507\Desktop\logsitic\horseColicTest.txt"
fileTrain=r"C:\Users\E507\Desktop\logsitic\horseColicTraining.txt"
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
#=====================
# 读取训练集
print("===== 读取训练集 =====")
X_train, y_train = load_dataset(fileTrain)
if X_train is None:
    exit()

# 读取测试集
print("\n===== 读取测试集 =====")
X_test, y_test = load_dataset(fileText)
if X_test is None:
    exit()

# 缺失值处理
print("\n===== 处理训练集缺失值 =====")
X_train_processed = replace_nan_with_mean(X_train)
print("\n===== 处理测试集缺失值 =====")
X_test_processed = replace_nan_with_mean(X_test)



#=====================
# 4. 构建并训练逻辑回归模型
#=====================
print("\n===== 训练逻辑回归模型 =====")
# 创建逻辑回归模型
lr_model = LogisticRegression(
    random_state=42,  # 随机种子，保证结果可复现
    max_iter=1000,    # 最大迭代次数
    solver='lbfgs'    # 优化器
)

# 训练模型
lr_model.fit(X_train_processed, y_train)

# 输出模型参数
print(f"模型系数（权重）：{lr_model.coef_}")
print(f"模型截距（偏置）：{lr_model.intercept_}")


#=====================
# 5. 测试集预测
#=====================
print("\n===== 测试集预测 =====")
# 预测类别
y_pred = lr_model.predict(X_test_processed)
# 预测概率
y_pred_proba = lr_model.predict_proba(X_test_processed)

print(f"前10个预测结果：{y_pred[:10]}")
print(f"前10个真实标签：{y_test[:10]}")
print(f"前10个预测概率：\n{y_pred_proba[:10]}")

#=====================
# 6. 计算准确率
#=====================

print("\n===== 模型评估 =====")
# 计算整体准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率：{accuracy:.4f} ({accuracy*100:.2f}%)")

# 计算训练集准确率（用于对比）
train_accuracy = accuracy_score(y_train, lr_model.predict(X_train_processed))
print(f"训练集准确率：{train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

# 额外：计算各类别的准确率
unique_classes = np.unique(y_test)
for cls in unique_classes:
    # 找到该类别的所有样本
    cls_indices = y_test == cls
    # 计算该类别的准确率
    cls_accuracy = accuracy_score(y_test[cls_indices], y_pred[cls_indices])
    print(f"类别 {cls} 的准确率：{cls_accuracy:.4f}")