import numpy as np
from sklearn.linear_model import LogisticRegression


#filename=r"c:\Users\Administrator\Desktop\机器学习\lesson4\testSet.txt"
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
train_dir = r"C:\Users\E507\Desktop\logsitic\horseColicTraining.txt"
X_train, y_train = load_dataset(train_dir)
# 读取测试集
test_dir  = r"C:\Users\E507\Desktop\logsitic\horseColicTest.txt" 
X_test,  y_test  = load_dataset(test_dir)
# 处理缺失值（重要：数据集可能存在缺失值）
X_train = replace_nan_with_mean(X_train)
X_test = replace_nan_with_mean(X_test)
#=====================
# 4. 构建并训练逻辑回归模型
#=====================
clf=LogisticRegression(solver="saga",max_iter=1000)
clf.fit(X_train,y_train)
#=====================
# 5. 测试集预测
#=====================
pred=clf.predict(X_test)
#=====================
# 6. 计算准确率
#=====================
acc=np.mean(pred==y_test)
print(acc)