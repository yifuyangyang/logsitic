import numpy as np
from sklearn.linear_model import LogisticRegression


filename=r"C:\Users\E507\Desktop\logsitic\horseColicTraining.txt"
filename1=r"C:\Users\E507\Desktop\logsitic\horseColicTest.txt"
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

def sigmoid(z):
    """补全sigmoid函数（原代码调用但未定义）"""
    return 1.0 / (1.0 + np.exp(-z))

def gradient_descent_logistic(X, y, lr=0.001, n_iters=8000):
    """补全梯度下降函数（原代码调用但未定义）"""
    # 保留原代码逻辑，仅适配维度（添加偏置项）
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 适配原代码预测时的维度
    n_samples, n_features = X_b.shape
    w = np.zeros(n_features)  # 初始化权重（含偏置项）
    # 批量梯度下降核心逻辑
    for _ in range(n_iters):
        z = np.dot(X_b, w)
        y_pred = sigmoid(z)
        gradient = (1 / n_samples) * np.dot(X_b.T, (y_pred - y))
        w -= lr * gradient
    return w

#=====================
# 3. 主流程
#=====================
train_x,train_y = load_dataset(filename)
test_x,test_y = load_dataset(filename1)

clf = LogisticRegression(solver="saga",max_iter=8000)
clf.fit(train_x,train_y)

pre=clf.predict(test_x)
acc=np.mean(pre==test_y)
print(acc)