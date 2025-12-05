import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 


fileText=r"C:\Users\E507\Desktop\logsitic\horseColicTest.txt"
fileTrain=r"C:\Users\E507\Desktop\logsitic\horseColicTraining.txt"
#=====================
# 1. 数据读取函数
#   （注意：若原始数据有?，需先替换为0）
#=====================
def load_dataset(filename):
    try:
        data = np.loadtxt(filename)
        X = data[:, :-1]   # 特征
        y = data[:, -1]    # 标签
        print(f"成功读取 {filename}，样本数：{X.shape[0]}，特征数：{X.shape[1]}")
        return X, y
    except Exception as e:
        print(f"读取数据失败：{e}")
        return None, None

#=====================
# 2. 缺失值处理函数
#   （将0替换为该列非0值的均值）
#=====================
def replace_nan_with_mean(X):
    X_processed = X.copy()  # 避免修改原数组
    for i in range(X_processed.shape[1]):
        col = X_processed[:, i]
        valid = col[col != 0]
        if len(valid) > 0:
            mean_val = np.mean(valid)
            col[col == 0] = mean_val
            X_processed[:, i] = col
        else:
            print(f"第{i+1}列全为0，无法计算均值，跳过")
    return X_processed

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
lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    solver='lbfgs'
)
lr_model.fit(X_train_processed, y_train)
print(f"模型系数（权重）：{lr_model.coef_}")
print(f"模型截距（偏置）：{lr_model.intercept_}")

#=====================
# 5. 预测与评估
#=====================
print("\n===== 测试集预测 =====")
y_pred = lr_model.predict(X_test_processed)
y_pred_proba = lr_model.predict_proba(X_test_processed)
print(f"前10个预测结果：{y_pred[:10]}")
print(f"前10个真实标签：{y_test[:10]}")
print(f"前10个预测概率：\n{y_pred_proba[:10]}")

print("\n===== 模型评估 =====")
test_acc = accuracy_score(y_test, y_pred)
train_acc = accuracy_score(y_train, lr_model.predict(X_train_processed))
print(f"测试集准确率：{test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"训练集准确率：{train_acc:.4f} ({train_acc*100:.2f}%)")

# 各类别准确率（增加鲁棒性）
unique_classes = np.unique(y_test)
for cls in unique_classes:
    cls_indices = y_test == cls
    if np.sum(cls_indices) == 0:
        print(f"类别 {cls} 无测试样本，跳过")
        continue
    cls_acc = accuracy_score(y_test[cls_indices], y_pred[cls_indices])
    print(f"类别 {cls} 的准确率：{cls_acc:.4f}")