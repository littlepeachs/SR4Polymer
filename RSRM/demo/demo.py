from itertools import compress
import numpy as np
from pandas import read_csv
import sys
sys.path.append("../")
from model.config import Config
from model.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sympy import symbols, parse_expr, Mul, Pow, S

complexity={
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
    "X":1,
    "exp1":2,
}

def preorder_traversal(expr, symbols=None):
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
            result.extend(preorder_traversal(arg, symbols))
        return result



def is_pareto_efficient(costs):
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
    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask

def pipeline(csv_1, csv_2=None):
    """
    Expression generation function in demo, input csv address, output the best expression and error by RSRM.
    :param csv_1: train csv file split by comma
    :param csv_2: test csv file split by comma
    :param const: True for using parameters with parameter optimization, False vice versa
    """
    if csv_2 is None:
        data = read_csv(csv_1, header=None)
        data = data.drop(data.index[0])
        x, t =np.array(data).T[:-1].T.astype(np.float64),np.array(data).T[-1].T.astype(np.float64)
        importance_index = [0,2,3,8]
        x = x[:,importance_index]
        x, x_test, t, t_test = train_test_split(x, t , test_size=0.2)

        x, t = x.T, t.T
        x_test, t_test = x_test.T, t_test.T
    else:
        csv1, csv2 = read_csv(csv_1, header=None), read_csv(csv_2, header=None)
        x, t = np.array(csv1).T[:-1], np.array(csv1).T[-1]
        x_test, t_test = np.array(csv2).T[:-1], np.array(csv2).T[-1]
    import pdb; pdb.set_trace()
    config = Config()
    config.json("../config/config.json")
    config.set_input(x=x, t=t, x_=x_test, t_=t_test)
    model = Pipeline(config=config)
    exp_cost_list = model.fit()
    costs = []
    if exp_cost_list[0][0]==None:
        exp_cost_list.pop(0)
    
    best_exp_cost_list=[]
    for p in exp_cost_list:
        cost=0
        p_expr= str(p[0]).lower()
        sympy_expr = parse_expr(str(p_expr))
        
        # 生成前序遍历序列
        preorder_sequence = preorder_traversal(sympy_expr)
        for token in preorder_sequence:
            if isinstance(token, (float,int)):
                cost += 1
            elif token[0]=="x" or token[0]=="X":
                cost += 1
            else:
                if token.lower() not in complexity:
                    cost+=1
                    print(f"Token {token} not in complexity dictionary")
                cost += complexity[token.lower()]
                
        rmse = p[1]
        inv_rmse = 1/(1+rmse)
        costs.append([cost, -inv_rmse])
        # if rmse<10:
        best_exp_cost_list.append([p_expr, rmse, cost])
    

    costs = np.array(costs)
    pareto_efficient_mask = is_pareto_efficient(costs)  # List of bool
    pf = list(compress(best_exp_cost_list, pareto_efficient_mask))
    pf.sort(key=lambda p: p[2]) # Sort by complexity
    import pdb; pdb.set_trace() 

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
    # print(f'\nresult: {best_exp} {rmse}')


if __name__ == '__main__':
    # pipeline("../data/nguyen/11_train.csv", "../data/nguyen/11_test.csv")
    # task_name = ["plate_height","avrami","heat_flow"]
    # task_type = ["accurate","high_var","middle_var","low_var","noise_0.02",\
    #             "noise_0.05","noise_0.1","noise_0.2","noise_0.5",\
    #             "num_10","num_25","num_100","num_200","num_500"]
    task_name = ["polymer_aging"] #"accurate","high_var","heat_flow", "accurate","high_var",
    task_type = ["accuracy"]
    print("############ NEW START ############")
    for task in task_name:
        for task_t in task_type:
            print("############ TASK ############")
            print("Task: ", task)
            print("Task type: ", task_t)
            print("############ END ############")
            # pipeline(f"../polymer_formula/{task_t}/{task}.csv")
            pipeline(f"../polymer_formula/aging_mirco_macro.csv")
            

