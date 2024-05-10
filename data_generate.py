import numpy as np
import pandas as pd
from math import exp,log,sin,cos,tan,pi

def get_formula(formula_name, varriance=None):
    if formula_name == 'thermal_noise':
        formula = '1 / (2 * pi * c) * (k / mu) ** 0.5'
        var_ranges = {'c': (0, 5), 'k': (0, 5), 'mu': (0, 5)}

    elif formula_name == 'larmor_frequency':
        formula = 'gamma * B0 / (2 * pi)'
        var_ranges = {'frequency': (0, 5), 'gamma': (0, 5), 'B0': (0, 5)} 

    elif formula_name == 'molecular_weight':
        formula = 'A - B * VR'
        var_ranges = {'log_M': (0, 5), 'A': (0, 5), 'B': (0, 5), 'VR': (0, 5)}

    elif formula_name == 'theoretical_plates':
        formula = '16 * (tR / w) ** 2'
        var_ranges = {'N': (0, 5), 'tR': (0, 5), 'w': (0, 5)}

    elif formula_name == 'plate_height':
        formula = 'L / N'
        var_ranges = {'L': (0, 2), 'N': (0, 2)}
        if varriance=="low":
            var_ranges = {'L': (0, 0.2), 'N': (0, 2)}
        elif varriance=="middle":
            var_ranges = {'L': (0, 0.02), 'N': (0, 2)}
        elif varriance=="high":
            var_ranges = {'L': (0, 0.002), 'N': (0, 2)}    

    elif formula_name == 'resolution':
        formula = '2 * (tR2 - tR1) / (w1 + w2)'
        var_ranges = {'Rs': (0, 5), 'tR1': (0, 5), 'tR2': (0, 5), 'w1': (0, 5), 'w2': (0, 5)}

    elif formula_name == 'heat_flow':
        formula = 'mass * heat_capacity * heating_rate'
        var_ranges = {'mass': (0, 2), 'heat_capacity': (0, 2), 'heating_rate': (0, 2)}
        if varriance=="low":
            var_ranges = {'mass': (0, 0.2), 'heat_capacity': (0, 2), 'heating_rate': (0, 20)}
        elif varriance=="middle":
            var_ranges = {'mass': (0, 0.02), 'heat_capacity': (0, 2), 'heating_rate': (0, 200)}
        elif varriance=="high":
            var_ranges = {'mass': (0, 0.002), 'heat_capacity': (0, 2), 'heating_rate': (0, 2000)}

    elif formula_name == 'bragg':
        formula = 'd = n * lambda/(2*sin(theta))'
        var_ranges = {
            'd': (0, 5),  # 晶面间距通常在0.1到10埃量级
            'theta': (0, 5),  # 衍射角通常小于30°,即0.5弧度
            'n': (0, 5),  # 衍射级数通常为小的正整数,如1,2,3,4,5
            'lambda': (0, 5)  # X射线波长通常在0.01到10埃量级
        }
    elif formula_name == 'hardness':
        formula = '100 - h / 0.002'
        var_ranges = {
            'h': (0.1, 2)  # 压入深度通常在0.1到2毫米之间
        }
    elif formula_name == 'chain_dimension':
        formula = 'R2 = N * l**2 * (1 + cos(theta)) / (1 - cos(theta))'
        var_ranges = {
        'N': (100,1000), # 链的节数通常在10到1000之间
        'l': (100,200), # 链节的长度通常在埃量级,如0.1到10埃
        'theta': (0.01*pi,pi) # 链节间的夹角theta在0到pi弧度之间
        }
        if varriance=="low":
            var_ranges = {
                'N': (100,1000), # 链的节数通常在10到1000之间
                'l': (10,20), # 链节的长度通常在埃量级,如0.1到10埃
                'theta': (0.01*pi,pi) # 链节间的夹角theta在0到pi弧度之间
                }
        elif varriance=="middle":
            var_ranges = {
                'N': (100,1000), # 链的节数通常在10到1000之间
                'l': (1,2), # 链节的长度通常在埃量级,如0.1到10埃
                'theta': (0.01*pi,pi) # 链节间的夹角theta在0到pi弧度之间
                }
        elif varriance=="high":
            var_ranges = {
                'N': (100,1000), # 链的节数通常在10到1000之间
                'l': (0.1,0.2), # 链节的长度通常在埃量级,如0.1到10埃
                'theta': (0.01*pi,pi) # 链节间的夹角theta在0到pi弧度之间
                }
    elif formula_name == 'avrami':
        formula = '1-exp(-k  * t**n)'
        var_ranges = {
        'k': (0,1), # 速率常数的量级可从每秒10^-6到1不等,取决于结晶条件
        # n是整数
        'n': range(1,5), # Avrami指数通常在1到4之间,取决于结晶机理
        't': (0,1) # 时间可从秒到天不等,取决于观察的时间尺度
        }
        if varriance=="low":
            var_ranges = {
            'k': (0,1), # 速率常数的量级可从每秒10^-6到1不等,取决于结晶条件
            # n是整数
            'n': range(1,5), # Avrami指数通常在1到4之间,取决于结晶机理
            't': (0,10) # 时间可从秒到天不等,取决于观察的时间尺度
            }
        elif varriance=="middle":
            var_ranges = {
            'k': (0,1), # 速率常数的量级可从每秒10^-6到1不等,取决于结晶条件
            # n是整数
            'n': range(1,5), # Avrami指数通常在1到4之间,取决于结晶机理
            't': (0,100) # 时间可从秒到天不等,取决于观察的时间尺度
            }
        elif varriance=="high":
            var_ranges = {
            'k': (0,1), # 速率常数的量级可从每秒10^-6到1不等,取决于结晶条件
            # n是整数
            'n': range(1,5), # Avrami指数通常在1到4之间,取决于结晶机理
            't': (0,1000) # 时间可从秒到天不等,取决于观察的时间尺度
            }
    return formula, var_ranges,formula_name

def generate_data_from_formula(formula, var_ranges, num_points, noise_level,y_name,formula_name,varriance=None):
    """
    根据给定的科学公式生成一系列随机分布且带有噪声的数值数据点,并将其存储到CSV文件中。

    参数:
    formula (str): 科学公式的字符串表示,其中自变量使用'x1', 'x2', ..., 'xn'表示。
    var_ranges (dict): 自变量的取值范围,格式为{'x1': (start1, end1), 'x2': (start2, end2), ...}。
    num_points (int): 要生成的总数据点数量。
    noise_level (float): 噪声水平,控制噪声的大小。默认为0.1。

    输出:
    将生成的数据点存储到名为'data.csv'的文件中,每行对应一组数据点。
    """
    var_names = list(var_ranges.keys())

    data = {var: [] for var in var_names}
    data['y'] = []

    for _ in range(num_points):
        point = {}
        for var in var_names:
            try:
                start, end = var_ranges[var]
                value = np.random.uniform(start, end)
            except ValueError:
                # 不是区间形式，可能是一个离散的自变量
                value = np.random.choice(var_ranges[var])
            
            point[var] = value
            data[var].append(value)
        locals().update(point)
        y = eval(formula)
        
        # 加入噪声
        noise = np.random.normal(0, noise_level * abs(y))
        y_with_noise = y + noise
        
        data['y'].append(y_with_noise)

    df = pd.DataFrame(data)
    if noise_level <= 0.01 and num_points==50 and varriance==None:
        df.to_csv(f'./polymer_formula/accurate/{formula_name}.csv', index=False)
    elif noise_level > 0.01 and num_points==50 and varriance==None:
        df.to_csv(f'./polymer_formula/noise_{str(noise_level)}/{formula_name}.csv', index=False)
    elif noise_level <= 0.01 and varriance==None and num_points!=50:
        df.to_csv(f'./polymer_formula/num_{str(num_points)}/{formula_name}.csv', index=False)
    elif noise_level <= 0.01 and num_points==50 and varriance!=None:
        df.to_csv(f'./polymer_formula/{str(varriance)}_var/{formula_name}.csv', index=False)
    print(f"已将{num_points}个随机分布且带有噪声的数据点存储到'data.csv'文件中。")


all_params = [(50,0.0,None),(50,0.02,None),(50,0.05,None),(50,0.1,None),
              (100,0.0,None),(200,0.0,None),(500,0.0,None),
              (50,0.0,"low"),(50,0.0,"middle"),(50,0.0,"high")]
formula_list = ['plate_height','heat_flow','chain_dimension','avrami']
for formula_name in formula_list:
    for params in all_params:
        num_points, noise_level,varriance = params
        formula, var_ranges,formula_name = get_formula(formula_name,varriance)
        final_formula = formula.split('=')[-1].strip()
        y_name = formula.split('=')[0].strip()
        generate_data_from_formula(final_formula, var_ranges, num_points, noise_level,y_name,formula_name,varriance)