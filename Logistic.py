#梯度上升算法测试函数
#求函数f(x) = -x^2 + 4x的极大值
def Gradient_Ascent_test():
    # f(x)的导数
    def f_prime(x_old):
        return -2 * x_old + 4
    # 初始值，给一个小于x_new的值
    x_old = -1
    #梯度上升算法初始值，即从(0,0)开始
    x_new = 0
    #梯度上升算法初始值，即从(0,0)开始
    # 步长，也就是学习速率，控制更新的幅度
    alpha = 0.01
    # 精度，也就是更新阈值
    presision = 0.00000001
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
        # 输出最终求解的极值近似值
    print(x_new)

if __name__ == '__main__':
    Gradient_Ascent_test()
