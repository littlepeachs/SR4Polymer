import re

def get_symbol_dict(num_symbols):
    symbol_dict = {}
    symbol_dict[0] = []
    for i in range(num_symbols):
        symbol_dict[0].append(f"x{i+1}")
    # for i in range(num_symbols):
        # symbol_dict[0].append(f"log(x{i+1})")
    for i in range(num_symbols):
        symbol_dict[0].append(f"exp(x{i+1})")
    # for i in range(num_symbols):
    #     symbol_dict[0].append(f"sqrt(x{i+1})")
    for i in range(num_symbols):
        symbol_dict[0].append(f"1/(x{i+1})")
    for i in range(num_symbols):
        for j in range(num_symbols):
            symbol_dict[0].append(f"x{i+1}*x{j+1}")

    symbol_dict[1] = []
    for i in range(num_symbols):
        for j in range(num_symbols):
            symbol_dict[1].append(f"x{i+1}*x{j+1}")
    for i in range(num_symbols):
        symbol_dict[1].append(f"(x{i+1})**(3/2)")
    for i in range(num_symbols):
        for j in range(num_symbols):
            symbol_dict[1].append(f"1/(x{i+1}*x{j+1})")
    for i in range(num_symbols):
        for j in range(num_symbols):
            symbol_dict[1].append(f"log(x{i+1})*log(x{j+1})")
    # for i in range(num_symbols):
    #     for j in range(num_symbols):
    #         symbol_dict[1].append(f"exp(x{i+1})*exp(x{j+1})")
    for i in range(num_symbols):
        for j in range(num_symbols):
            symbol_dict[1].append(f"log(x{i+1})/x{j+1}")
    for i in range(num_symbols):
        for j in range(num_symbols):
            symbol_dict[1].append(f"exp(x{i+1})/x{j+1}")
    for i in range(num_symbols):
        for j in range(num_symbols):
            if i!=j:
                symbol_dict[1].append(f"x{i+1}/x{j+1}")


    symbol_dict[2] = []
    for i in range(num_symbols):
        for j in range(num_symbols):
            for k in range(num_symbols):
                symbol_dict[2].append(f"x{i+1}*x{j+1}*x{k+1}")
    unique_expressions = set()

    for expr in symbol_dict[2]:
        # 对每个表达式的因子进行排序（字典序）
        factors = expr.split('*')
        sorted_factors = sorted(factors)
        sorted_expr = '*'.join(sorted_factors)
        # 将排序后的表达式添加到集合中
        unique_expressions.add(sorted_expr)

    # 转换回列表并排序
    unique_expressions_list = sorted(list(unique_expressions))
    symbol_dict[2] = unique_expressions_list

    expressions = []
    for i in range(num_symbols):
        for j in range(num_symbols):
            for k in range(num_symbols):
                expressions.append(f"1/(x{i+1}*x{j+1}*x{k+1})")
    unique_expressions = set()

    for expr in expressions:
        # 分离分子和分母
        numerator, denominator = expr.split('/')
        # 对分母表达式的因子进行排序
        factors = denominator.strip('()').split('*')
        sorted_factors = sorted(factors)
        # 重新构建排序后的分数表达式
        sorted_denominator = '*'.join(sorted_factors)
        sorted_expr = f"1/({sorted_denominator})"
        # 将排序后的表达式添加到集合中
        unique_expressions.add(sorted_expr)

    # 转换回列表并排序
    unique_expressions_list = sorted(list(unique_expressions))
    symbol_dict[2].extend(unique_expressions_list)

    expressions = []
    for i in range(num_symbols):
        for j in range(num_symbols):
            for k in range(num_symbols):
                expressions.append(f"log(x{i+1})*log(x{j+1})*log(x{k+1})")
    unique_expressions = set()

    for expr in expressions:
        # 将表达式按'*'分割成单个对数因子
        factors = expr.split('*')
        # 对因子进行排序
        sorted_factors = sorted(factors)
        # 重新组合表达式
        sorted_expr = '*'.join(sorted_factors)
        # 添加到集合中去重
        unique_expressions.add(sorted_expr)

    # 转换回列表并排序
    unique_expressions_list = sorted(list(unique_expressions))
    symbol_dict[2].extend(unique_expressions_list)

    expressions = []
    for i in range(num_symbols):
        for j in range(i,num_symbols):
            for k in range(num_symbols):
                expressions.append(f"log(x{i+1})*log(x{j+1})/(x{k+1})")
    symbol_dict[2].extend(expressions)

    expressions = []
    for i in range(num_symbols):
        for j in range(num_symbols):
            for k in range(j,num_symbols):
                expressions.append(f"log(x{i+1})/(x{j+1}*x{k+1})")
    symbol_dict[2].extend(expressions)

    expressions = []
    for i in range(num_symbols):
        for j in range(num_symbols):
            for k in range(j,num_symbols):
                expressions.append(f"sqrt(x{i+1})*x{j+1}*x{k+1}")
    symbol_dict[2].extend(expressions)

    return symbol_dict

if __name__ == "__main__":
    symbol_dict = get_symbol_dict(9)

    print(len(symbol_dict[0]), len(symbol_dict[1]), len(symbol_dict[2]))


# print(symbol_dict)