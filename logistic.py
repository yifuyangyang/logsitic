import numpy as np
from sklearn.linear_model import LogisticRegression

filename=r"C:\Users\E507\Desktop\logistic\logsitic\testSet.txt"
train_filename = r"C:\Users\E507\Desktop\logistic\logsitic\horseColicTraining.txt"  
test_filename = r"C:\Users\E507\Desktop\logistic\logsitic\horseColicTest.txt"  

def load_dataset(filename):
    """读取数据集文件"""
    try:
        data = np.loadtxt(filename)
        X = data[:, :-1]   # 特征
        y = data[:, -1]    # 标签
        print(f"成功读取数据：{filename}，特征维度{X.shape}，标签数量{y.shape[0]}")
        return X, y
    except FileNotFoundError:
        print(f"错误：文件 {filename} 未找到！")
        return None, None
    except Exception as e:
        print(f"读取数据出错：{e}")
        return None, None

def replace_nan_with_mean(X):
    """将特征矩阵中的0值（缺失值）替换为对应列的非0值均值"""
    X_processed = X.copy()  # 避免修改原数据
    for i in range(X_processed.shape[1]):
        col = X_processed[:, i]
        valid = col[col != 0]  # 非0值为有效数据
        if len(valid) > 0:
            mean_val = np.mean(valid)
            col[col == 0] = mean_val
            X_processed[:, i] = col
    return X_processed

# =====================
# 3. 读取并校验数据
# =====================
# 读取训练集
X_train, y_train = load_dataset(train_filename)
# 读取测试集
X_test, y_test = load_dataset(test_filename)

# 数据有效性检查
if X_train is None or X_test is None or len(y_train) == 0 or len(y_test) == 0:
    raise ValueError("数据读取失败，请检查文件路径和文件格式")

# =====================
# 4. 构建并训练逻辑回归模型
# =====================
X_train_processed = replace_nan_with_mean(X_train)
clf = LogisticRegression(max_iter=1000, random_state=42)  
clf.fit(X_train_processed, y_train)
print("模型训练完成")

# =====================
# 5. 测试集预测
# =====================
# 对测试数据进行缺失值处理
X_test_processed = replace_nan_with_mean(X_test)
# 测试集预测
pred = clf.predict(X_test_processed)

# =====================
# 6. 计算准确率
# =====================
acc = np.mean(pred == y_test)  # 用测试集真实标签计算准确率
# 打印准确率
print(f"\n测试集预测准确率：{acc:.4f} ({acc*100:.2f}%)")

