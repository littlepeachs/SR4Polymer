import numpy as np
from sympy import symbols, lambdify, sin, cos, exp, log, Abs
from sklearn.linear_model import Lasso, ElasticNet

# 生成示例数据
np.random.seed(0)
X = np.random.uniform(2, 100, size=(100, 2))
y = 3 * X[:, 1] + 1

# 定义候选符号
x1, x2 = symbols('x1 x2')
candidate_symbols = [x1, x2, x1**2, x2**2, x1*x2, sin(x1), cos(x2), exp(x1), log(Abs(x2) + 1)]

# 特征工程
features = []
for symbol in candidate_symbols:
    func = lambdify([x1, x2], symbol, 'numpy')
    feature = func(X[:, 0], X[:, 1])
    features.append(feature)
features = np.array(features).T

# Lasso 回归
lasso = Lasso(alpha=0.5)
lasso.fit(features, y)

# Clip step
cutoff = 1e-2  # 设置一个阈值
lasso_coefs = lasso.coef_.copy()
lasso_coefs[np.abs(lasso_coefs) < cutoff] = 0

# ElasticNet 回归
enet = ElasticNet(alpha=0.5, l1_ratio=0.5)
enet.fit(features, y)

# Clip step II
enet_coefs = enet.coef_.copy()
enet_coefs[np.abs(enet_coefs) < cutoff] = 0

# 解释结果
print("Lasso 回归表达式:")
lasso_expression = f"{lasso.intercept_:.2f}"
for coef, symbol in zip(lasso_coefs, candidate_symbols):
    if coef != 0:
        lasso_expression += f" + {coef:.2f} * {symbol}"
print(lasso_expression)

print("\nElasticNet 回归表达式:")
enet_expression = f"{enet.intercept_:.2f}"
for coef, symbol in zip(enet_coefs, candidate_symbols):
    if coef != 0:
        enet_expression += f" + {coef:.2f} * {symbol}"
print(enet_expression)