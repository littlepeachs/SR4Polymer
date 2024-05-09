import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem

# 定义问题
class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=1,                  # 问题的维度
                         n_obj=2,                  # 目标函数的数量
                         n_constr=0,               # 约束条件的数量
                         xl=np.array([-10]),       # 变量的下界
                         xu=np.array([10]))        # 变量的上界

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0]**2
        f2 = (x[0]-2)**2
        out["F"] = [f1, f2]

# 使用帕累托优化算法
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter

algorithm = NSGA2(pop_size=40)

problem = MyProblem()

res = minimize(problem,
               algorithm,
               ('n_gen', 40),
               verbose=False,
               seed=1)

# 结果可视化
plot = Scatter()
plot.add(res.F, color="red")
plot.show()
plot.save('pareto.png')