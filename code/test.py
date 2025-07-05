def calculate_fibonacci(n):
    """计算斐波那契数列的前n项"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib

# 生成前20项斐波那契数列
fibonacci_sequence = calculate_fibonacci(20)
print("斐波那契数列前20项：", fibonacci_sequence)
