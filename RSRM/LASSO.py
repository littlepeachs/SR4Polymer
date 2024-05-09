import numpy as np
from sympy import symbols, lambdify, sin, cos, exp,log, parse_expr
from sklearn.linear_model import Lasso
from scipy.signal import savgol_filter
import numpy as np
from symbol_generate import get_symbol_dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pandas import read_csv
window_size = 5  # 滤波器窗口大小（必须为奇数）
poly_order = 2    # 多项式的阶数
degree = 0       # 多项式的阶数
clip=1e-4
clip_level=4
proj_dict = {"exp": np.exp,"log": np.log,"sqrt": np.sqrt}
train_proj_dict = {"exp": np.exp,"log": np.log,"sqrt": np.sqrt}

task_name = ["polymer_aging"] #"accurate","high_var","heat_flow", "accurate","high_var",
task_type = ["accuracy"]
    # 生成示例数据
np.random.seed(0)
for task in task_name:
    for task_t in task_type:
        print("############ TASK ############")
        print("Task: ", task)
        print("Task type: ", task_t)
        print("############ END ############")
        if task=="plate_height":
            num_symbols=2
        elif task=="avrami" or task=="heat_flow":
            num_symbols=3
        else:
            num_symbols=9
        data = read_csv(f"./polymer_formula/aging_mirco_macro.csv", header=None)

        data = data.drop(data.index[0])
        X, y = np.array(data).T[:-1].T.astype(np.float64), np.array(data).T[-1].T.astype(np.float64)
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

        X_list = [X[:,i] for i in range(X.shape[1])]
        X_test_list = [X_test[:,i] for i in range(X_test.shape[1])]
        proj_dict.update({f'x{i+1}': X_test[:,i] for i in range(X_test.shape[1])})
        train_proj_dict.update({f'x{i+1}': X[:,i] for i in range(X.shape[1])})
        # 定义候选符号,包括常数项1
        original_string = ""
        for i in range(num_symbols):
            original_string += f"x{i+1} "
        symbol_x= symbols(original_string)

        symbols_dict = get_symbol_dict(num_symbols)
        # import pdb;pdb.set_trace()
        candidate_symbols = []
        for i in range(degree+1):
            candidate_symbols.extend(symbols_dict[i])
        # 特征工程
        features = []
        for symbol in candidate_symbols:
            func = lambdify([*symbol_x], symbol, 'numpy')
            feature = func(*X_list)
            features.append(feature)
        features = np.array(features).T

        # Lasso 回归
        lasso = Lasso(alpha=2)
        lasso.fit(features, y)


        # 解释结果
        print("回归表达式:")
        expression = f"{lasso.intercept_:.2f}"
        clip_expression = f"{lasso.intercept_:.2f}"
        for coef, symbol in zip(lasso.coef_, candidate_symbols):
            if coef != 0:
                expression += f" + {coef} * {symbol}"
            if coef >= clip:
                clip_expression += f" + {coef:.{clip_level}f} * {symbol}"
        print("#### expression",expression)
        print("#### clip_expression",clip_expression)
        y_pred_clip = eval(clip_expression, proj_dict)
        y_train_pred_clip = eval(clip_expression, train_proj_dict)

        # TEST
        test_features = []
        for symbol in candidate_symbols:
            func = lambdify([*symbol_x], symbol, 'numpy')
            feature = func(*X_test_list)
            test_features.append(feature)
        test_features = np.array(test_features).T
        y_pred = lasso.predict(test_features)
        y_train_pred = lasso.predict(features)

        if not isinstance(y_pred, (list, np.ndarray)):
            y_pred = [y_pred] * len(y_test)
        if not isinstance(y_pred_clip, (list, np.ndarray)):
            y_pred_clip = [y_pred_clip] * len(y_test)
        if not isinstance(y_train_pred, (list, np.ndarray)):
            y_train_pred = [y_train_pred] * len(y)
        if not isinstance(y_train_pred_clip, (list, np.ndarray)):
            y_train_pred_clip = [y_train_pred_clip] * len(y)
        # 计算 RMSE 和 R^2
        # import pdb;pdb.set_trace()
        print("#### origin result")
        rmse = np.sqrt(mean_squared_error(y, y_train_pred))
        r2 = r2_score(y, y_train_pred)
        print(f"Train RMSE: {rmse:.5f}")
        print(f"Train R^2: {r2:.5f}")

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Valid RMSE: {rmse:.5f}")
        print(f"Valid R^2: {r2:.5f}")

        print("#### clip result")
        rmse = np.sqrt(mean_squared_error(y, y_train_pred_clip))
        r2 = r2_score(y, y_train_pred_clip)
        print(f"Train RMSE: {rmse:.5f}")
        print(f"Train R^2: {r2:.5f}")

        rmse = np.sqrt(mean_squared_error(y_test, y_pred_clip))
        r2 = r2_score(y_test, y_pred_clip)
        print(f"Valid RMSE: {rmse:.5f}")
        print(f"Valid R^2: {r2:.5f}")
        print("########### END #########")
        

