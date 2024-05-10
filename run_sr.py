import tensorflow as tf 
from dso import DeepSymbolicRegressor
import numpy as np
from pandas import read_csv
# Generate some data
np.random.seed(0)
import json
from scipy.ndimage import gaussian_filter1d
from itertools import compress
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import r2_score
from sympy import symbols, parse_expr, Mul, Pow, S
from math import exp,log,cos,sin
import argparse
import sys
sys.path.append("./RSRM/")
from model.config import Config
from model.pipeline import Pipeline




class Evaluator:
    def __init__(self):
        self.complexity={
            "add":1,
            "sub":1,
            "mul":1,
            "div":1,
            "sin":3,
            "cos":3,
            "tan":3,
            "exp":2,
            "log":2,
            "sqrt":3,
            "pow":2,
            "n2":2,
            "x":1,
            "exp1":2,
        }
        

    def preorder_traversal(self,expr, symbols=None):
        """
        递归生成表达式的前序遍历序列（列表形式）。
        :param expr: SymPy 表达式对象
        :param symbols: 表达式中使用的符号列表（可选）
        :return: 前序遍历序列的列表表示
        """
        if symbols is None:
            symbols = []

        # 如果是符号，返回符号的字符串表示
        if expr.is_Symbol:
            return [str(expr)]
        # 如果是数值，返回数值
        elif expr.is_Number:
            return [float(expr)]
        # 如果是其他类型的表达式，递归遍历
        else:
            # 访问根节点（算符）
            result = [expr.__class__.__name__]  # 添加算符
            # 遍历子节点
            for arg in expr.args:
                result.extend(self.preorder_traversal(arg, symbols))
            return result

    def is_pareto_efficient(self,costs):
        """
        Find the pareto-efficient points given an array of costs.

        Parameters
        ----------

        costs : np.ndarray
            Array of shape (n_points, n_costs).

        Returns
        -------

        is_efficient_maek : np.ndarray (dtype:bool)
            Array of which elements in costs are pareto-efficient.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index<len(costs):
            nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        is_efficient_mask = np.zeros(n_points, dtype=np.bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask

    def get_cost(self,all_programs,algo="DSR"):
        if "DSR" in algo:
            for p in all_programs:
                p.complexity=0
                p_expr= str(p.sympy_expr)
                sympy_expr = parse_expr(str(p_expr))

                # 生成前序遍历序列
                preorder_sequence = self.preorder_traversal(sympy_expr)
                for token in preorder_sequence:
                    if isinstance(token, (float,int)):
                        p.complexity += 1
                    elif token[0]=="x":
                        p.complexity += 1
                    else:
                        p.complexity += self.complexity[token.lower()]
            costs = np.array([(p.complexity, -p.r) for p in all_programs])
            pareto_efficient_mask = self.is_pareto_efficient(costs)  # List of bool
            pf = list(compress(all_programs, pareto_efficient_mask))
            pf.sort(key=lambda p: p.complexity) # Sort by complexity
            return pf,costs
        
        elif algo == "RSRM":
            costs = []
            best_exp_cost_list=[]
            for p in all_programs:
                cost=0
                p_expr= str(p[0]).lower()
                sympy_expr = parse_expr(str(p_expr))
                
                # 生成前序遍历序列
                preorder_sequence = self.preorder_traversal(sympy_expr)
                for token in preorder_sequence:
                    if isinstance(token, (float,int)):
                        cost += 1
                    elif token[0]=="x" or token[0]=="X":
                        cost += 1
                    else:
                        if token.lower() not in self.complexity:
                            cost+=1
                            print(f"Token {token} not in complexity dictionary")
                        cost += self.complexity[token.lower()]
                rmse = p[1]
                inv_rmse = 1/(1+rmse)
                costs.append([cost, -inv_rmse])
                # if rmse<10:
                best_exp_cost_list.append([p_expr, rmse, cost])

            costs = np.array(costs)
            pareto_efficient_mask = self.is_pareto_efficient(costs)  # List of bool
            pf = list(compress(best_exp_cost_list, pareto_efficient_mask))
            pf.sort(key=lambda p: p[2]) 
            return pf,costs
                    
                
    
def main():
    args = argparse.ArgumentParser()
    args.add_argument("--task", type=str, default="benchmark",choices=["benchmark","real"])
    args.add_argument("--split", type=str, default="random",choices=["cv","random"])
    args.add_argument("--algo", type=str, default="RSRM",choices=["DSR","uDSR","RSRM","KAN","LASSO"])
    args.add_argument("--model_config", type=str, default="./config/config_regression_pg.json")
    args = args.parse_args()
    if args.task=="real":
        task_name = ["polymer_aging"] #"accurate","high_var","heat_flow", "accurate","high_var",
        task_type = ["accuracy"]
    elif args.task=="benchmark":
        task_name = ["plate_height","heat_flow","avrami"]
        task_type = ["accurate","noise_0.02",\
                    "noise_0.05","noise_0.1","noise_0.2",\
                    "noise_0.5","low_var","middle_var","high_var"\
                    "num_10","num_25","num_100",\
                    "num_200","num_500"]
    else:
        raise NotImplementedError
    
    evaluator = Evaluator()
    
    print("############ NEW START ############")
    for task in task_name:
        for task_t in task_type:
            data = read_csv(f"./polymer_formula/{task_t}/{task}.csv", header=None)
            data = data.drop(data.index[0])
            X, y = np.array(data).T[:-1].T.astype(np.float64), np.array(data).T[-1].T.astype(np.float64)
            # importance_index = [0,2,3,8]
            # X = X[:,importance_index]
            import pdb; pdb.set_trace()
            cv = KFold(n_splits=5, random_state=42, shuffle=True)

            for train_index, test_index in cv.split(X):
                print("############ TASK ############")
                print("Task: ", task)
                print("Task type: ", task_t)
                print("############ END ############")

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # import pdb; pdb.set_trace()
                if args.split=="random":
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

                if args.algo in ["DSR","uDSR"]:
                    config = json.load(open(args.model_config))
                    model = DeepSymbolicRegressor(config) # Alternatively, you can pass in your own config JSON path
                    # Fit the model
                    model.fit(X_train, y_train) # Should solve in ~10 seconds
                    # View the best expression
                    all_programs = [model.program_[i]["program"] for i in range(len(model.program_))]
                    pf,costs = evaluator.get_cost(all_programs)
                    model.program_ = pf[0]

                    # Make predictions
                    valid_output = model.predict(X_test)
                    if len(valid_output.shape)!=1:
                        valid_output = valid_output[0]
                    # import pdb; pdb.set_trace()
                    valid_output = np.nan_to_num(valid_output, nan=0)
                    r2 = r2_score(y_test, valid_output)
                    rmse = np.sqrt(np.mean((valid_output - y_test) ** 2))

                    print("############ Pareto Boundary Program ############")
                    for p in pf:
                        p.print_stats()
                        print("Complexity: ", p.complexity)
                        print("Reward: ", p.r)
                        print("######### End Of This Expression #########")

                    print("############ RESULT ############",flush=True)
                    print("Task: ", task,flush=True)
                    print("Task type: ", task_t,flush=True)
                    print("Number of programs in the Pareto front: ", len(pf),flush=True)
                    print("Less Complexity pareto optimal programs:",flush=True)
                    model.program_.print_stats()
                    print("Complexity\tR2",flush=True)
                    print(costs,flush=True)
                    print("Validation R2: ", r2,flush=True)
                    print("Validation RMSE: ", rmse,flush=True)
                    print("############## END ##############",flush=True)
                
                elif args.algo=="RSRM":
                    X_train,X_test,y_train,y_test =X_train.T,X_test.T,y_train.T,y_test.T
                    config = Config()
                    config.json("./RSRM/config/config.json")
                    config.set_input(x=X_train, t=y_train, x_=X_test, t_=y_test)
                    model = Pipeline(config=config)
                    exp_cost_list = model.fit()
                    if exp_cost_list[0][0]==None:
                        exp_cost_list.pop(0)
                    pf,costs = evaluator.get_cost(exp_cost_list,algo="RSRM")
                    print("\n############ Pareto Boundary Program ############")
                    for p in pf:
                        print("Expression: ", p[0])
                        print("Complexity: ", p[2])
                        print("RMSE: ", p[1])
                        print("######### End Of This Expression #########")

                    print("############ RESULT ############",flush=True)
                    print("Task: ", task,flush=True)
                    print("Task type: ", task_t,flush=True)
                    print("Number of programs in the Pareto front: ", len(pf),flush=True)
                    print("Less Complexity pareto optimal programs:",flush=True)
                    print(pf[0][0])
                    print("Complexity\tR2",flush=True)
                    print(pf[0][2],flush=True)
                    print("Validation RMSE: ", pf[0][1],flush=True)
                    print("############## END ##############",flush=True)

                if args.split=="random":
                    break

if __name__=="__main__":
    main()