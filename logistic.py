import numpy as np
from sklearn.linear_model import LogisticRegression


#filename=r"C:\Users\E507\Desktop\logsitic\horseColicTest.txt"
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
if __name__ == "__main__":
        # 假设你有训练集和测试集文件
        train_filename = r"C:\Users\E507\Desktop\logsitic\horseColicTraining.txt"  
        # 训练集文件
        test_filename = r"C:\Users\E507\Desktop\logsitic\horseColicTest.txt"      
           
        # 读取训练集
        print("正在读取训练集...")
        train_X, train_y = load_dataset(train_filename)
        print(f"训练集形状: X={train_X.shape}, y={train_y.shape}")
        
        # 读取测试集
        print("正在读取测试集...")
        test_X, test_y = load_dataset(test_filename)
        print(f"测试集形状: X={test_X.shape}, y={test_y.shape}")
        
        # 处理缺失值（根据你的数据实际情况调整缺失值标记）
        print("正在处理缺失值...")
        train_X = replace_nan_with_mean(train_X)  
        test_X = replace_nan_with_mean(test_X)    
            #=====================
            # 4. 构建并训练逻辑回归模型
            #=====================
        print("正在训练逻辑回归模型...")
        clf = LogisticRegression(max_iter=1000,random_state=42)
        clf.fit(train_X, train_y)
            #=====================<
           # 5. 测试集预测
            #=====================<
        print("正在对测试集进行预测...")
        pred = clf.predict(test_X)
    
           #=====================
            # 6. 计算准确率
            #=====================
        accuracy = np.mean(pred == test_y)  
        print(f"模型准确率: {accuracy:.4f}")
            








