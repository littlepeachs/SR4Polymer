import numpy as np
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

# 定义可能的特征符号
candidate_symbols = ['x1', 'x2', 'x1**2', 'x2**2', 'x1*x2', 'sin(x1)', 'cos(x2)', 'exp(x1)', 'log(Abs(x2) + 1)']

import numpy as np

def generate_features(X, degree, intercept=True):
    """
    Generate features based on the degree.
    
    Parameters:
    - X: The input data with shape (n_samples, n_features).
    - degree: The degree of the polynomial features to generate.
    - intercept: Whether to include the intercept (constant term) in the features.
    
    Returns:
    - features: An array with the generated features.
    """
    # Initialize the list of features
    features = []

    # Add intercept if specified
    if intercept:
        features.append(np.ones((X.shape[0], 1)))
    
    # Add original features
    features.append(X)
    
    # Generate features based on the degree
    for i in range(1, degree + 1):
        # Add power features
        features.append(np.power(X, i))
        
        # Add square root features
        if i % 2 == 0:
            features.append(np.sqrt(X))
        
        # Add logarithmic features
        features.append(np.log(X + 1))  # To avoid log(0), add 1 to all elements
        
        # Add reciprocal features
        features.append(1 / X)
        
        # Add interaction features with previous degrees
        for j in range(1, i + 1):
            if i - j > 0:
                for k in range(X.shape[1]):
                    for l in range(k + 1, X.shape[1]):
                        features.append(X[:, k] ** (i - j) * X[:, l] ** j)
    
    # Concatenate all features and remove duplicates
    features = np.hstack(features)
    unique_features, unique_indices = np.unique(features, axis=1, return_inverse=True)
    features = features[:, unique_indices]

    return features

# 示例数据
X = np.random.rand(100, 5)  # 假设有100个样本和5个特征

# 指定度数为1
degree = 1

# 生成特征集
features = generate_features(X, degree, intercept=False)
print("Generated Features Shape:", features.shape)
import pdb; pdb.set_trace()

def lasso_clip_en(X, y, alphas, degrees, lags, cutoff):
    # LASSO步骤
    best_mse = np.inf
    best_params = None
    best_coefs = None
    for alpha in alphas:
        for lag in lags:
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(X, y)
            mse = -cross_val_score(lasso, X, y, cv=5, scoring='neg_mean_squared_error')
            if mse < best_mse:
                best_mse = mse
                best_params = (alpha, lag)
                best_coefs = lasso.coef_.copy()
    
    # Clip步骤
    if best_coefs is not None:
        best_coefs[best_coefs < cutoff] = 0
    
    # EN步骤
    best_en_mse = np.inf
    best_en_params = None
    for alpha in alphas:
        for l1_ratio in [0.1, 0.5, 0.7, 0.9]:
            en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
            # 使用经过Clip步骤的系数作为EN模型的初始值
            en.set_params(coef_=best_coefs)
            en.fit(X, y)
            mse = -cross_val_score(en, X, y, cv=5, scoring='neg_mean_squared_error')
            if mse < best_en_mse:
                best_en_mse = mse
                best_en_params = (alpha, l1_ratio)
    
    # 最终模型
    en_best = ElasticNet(alpha=best_en_params[0], l1_ratio=best_en_params[1], max_iter=10000)
    # 由于在EN步骤中已经使用了Clip步骤的系数，这里不需要再次Clip
    en_best.set_params(coef_=best_coefs)
    en_best.fit(X, y)
    
    return en_best, en_best.coef_


def run_lcen(X, y, alphas, degrees, lags, cutoff):
    # 生成特征
    features = generate_features(X, max(degrees))
    X_new = np.column_stack(features)
    
    # 运行LCEN算法
    model, coefficients = lasso_clip_en(X_new, y, alphas, degrees, lags, cutoff)
    
    return model, coefficients

# 示例数据
X = np.random.rand(100, 2)  # 假设有100个样本和2个特征
y = X[:,0]**2+X[:,1]     # 目标变量

# 超参数范围
alphas = [0.01, 0.1, 1]
degrees = [0, 1, 2]  # 特征复杂度
lags = [0, 1]
cutoff = 0.01

# 运行LCEN算法
model, coefficients = run_lcen(X, y, alphas, degrees, lags, cutoff)

# 输出结果
print("Final Model Coefficients:", coefficients)