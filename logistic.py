import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

filename=r"C:\Users\E507\Documents\GitHub\logsitic\test.txt"
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
if __name__ =="__main__":
    train_filename=r"C:\Users\E507\Desktop\horseColicTraining.txt"
    train_x,train_y=load_dataset(train_filename)
    test_filename=r"C:\Users\E507\Desktop\horseColicTest.txt"
    test_x,test_y=load_dataset(test_filename)

    train_x=replace_nan_with_mean(train_x)
    test_x=replace_nan_with_mean(test_x)
    model=LogisticRegression(random_state=42,max_iter=1000)
    model.fit(train_x,train_y)

    y_pred =model.predict(test_x)

    accuracy = accuracy_score(test_y,y_pred)
    print(f"测试集准确率：{accuracy:.4f}")
    print(f"测试集准确率百分比：{accuracy*100:.2f}%")

# 读取测试集


#=====================
# 4. 构建并训练逻辑回归模型
#=====================


#=====================
# 5. 测试集预测
#=====================


#=====================
# 6. 计算准确率
#=====================

