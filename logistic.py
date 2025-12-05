import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

filename=r"C:\Users\E507\Desktop\logistic\logsitic\testSet.txt"
train_filename = r"C:\Users\E507\Desktop\logistic\logsitic\horseColicTraining.txt"  
test_filename = r"C:\Users\E507\Desktop\logistic\logsitic\horseColicTest.txt"  
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
X_train, y_train = load_dataset(train_filename)
# 读取测试集
X_test, y_test = load_dataset(test_filename)
# 数据有效性检查
if X_train is None or X_test is None or len(y_train) == 0 or len(y_test) == 0:
    raise ValueError("数据读取失败，请检查文件路径和文件格式")
#=====================
# 4. 构建并训练逻辑回归模型
#=====================
def train_logistic_regression(X_train, y_train):
    """
    训练逻辑回归模型
    :param X_train: 训练集特征
    :param y_train: 训练集标签
    :return: 训练好的模型
    """
    # 构建模型（设置max_iter确保收敛，random_state保证结果可复现）
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    # 训练模型
    lr_model.fit(X_train, y_train)
    print("模型训练完成")
    return lr_model

# 对训练数据进行缺失值处理
X_train_processed = replace_nan_with_mean(X_train)
# 训练模型
model = train_logistic_regression(X_train_processed, y_train)
#=====================
# 5. 测试集预测
#=====================
def predict_with_model(model, X_test):
    """
    使用训练好的模型进行预测
    :param model: 训练好的模型
    :param X_test: 测试集特征
    :return: 预测结果
    """
    # 对测试数据进行缺失值处理
    X_test_processed = replace_nan_with_mean(X_test)
    # 预测
    y_pred = model.predict(X_test_processed)
    return y_pred, X_test_processed

# 执行预测
y_pred, X_test_processed = predict_with_model(model, X_test)
#=====================
# 6. 计算准确率
#=====================
def calculate_accuracy(y_true, y_pred):
    """
    计算预测准确率
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 准确率
    """
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

# 计算并打印准确率
accuracy = calculate_accuracy(y_test, y_pred)
print(f"\n测试集预测准确率：{accuracy:.4f} ({accuracy*100:.2f}%)")
