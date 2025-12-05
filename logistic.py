import numpy as np
from sklearn.linear_model import LogisticRegression


train_dir = r"C:\Users\86183\OneDrive\Desktop\logsitic\horseColicTraining.txt"  
test_dir  = r"C:\Users\86183\OneDrive\Desktop\logsitic\horseColicTest.txt"       
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


def main():
    X_train, y_train = load_dataset(train_dir)
    X_test,  y_test  = load_dataset(test_dir)
    # print("使用梯度下降训练Logistic回归")
    # w_gd=gradient_descent_logistic(x,y,lr=0.001,n_iters=8000)
    # print(f"\n[GD]Learned w={w_gd}")
    # print("\n使用sklearn LogisticRegression训练")
    clf=LogisticRegression(solver='lbfgs',max_iter=3000)
    clf.fit(X_train,y_train)
    # print(f"[sklearn coef_={clf.coef_},intercept_={clf.intercept_}]")
    # y_pred_gd=(sigmoid(np.c_[np.ones((x.shape[0],1)),x]@w_gd)>=0.5).astype(int)
    # acc_gd=np.mean(y_pred_gd==y)
    y_pred_sk=clf.predict(X_test)
    acc_sk=np.mean(y_pred_sk==y_test)
    # print(f"\n训练集准确率:")
    # print(f"自写GD Logistic 回归 accuracy={acc_gd:.4f}")
    # print(f"sklearn LogisticRegression accuracy={acc_sk:.4f}")
    # plot_compare_boundaries(x,y,w_gd,clf)
    print(acc_sk)
# 读取测试集
if __name__ == "__main__":
    main()

#=====================
# 4. 构建并训练逻辑回归模型
#=====================


#=====================
# 5. 测试集预测
#=====================


#=====================
# 6. 计算准确率
#=====================

