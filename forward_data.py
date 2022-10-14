import numpy as np
import math
import os


def data_merge(SNR, pure_echo_path, echo_noise_path, T2_AMPLITUDE_path, save_pure_echo_path, save_echo_noise_path, save_T2_AMPLITUDE_path):
    # 无噪声回波串
    pure_echo_list = []
    # 有噪声回波串
    echo_noise_list = []
    # 幅度
    T2_AMPLITUDE_list = []
    for root, dirs, files in os.walk(pure_echo_path):
        for file in files:
            pure_echo = np.load(pure_echo_path + file)
            pure_echo_list.append(list(pure_echo))
    pure_echo_list = np.array(pure_echo_list).reshape(-1, 2500)
    np.save(save_pure_echo_path + 'pure_echo'+'_SNR' + str(SNR) + '.npy', pure_echo_list)
    pure_echo_list = []

    for root, dirs, files in os.walk(echo_noise_path):
        for file in files:
            echo_noies = np.load(echo_noise_path + file)
            echo_noise_list.append(list(echo_noies))
    echo_noise_list = np.array(echo_noise_list).reshape(-1, 2500)
    np.save(save_echo_noise_path + 'echo_noise' + '_SNR' + str(SNR) + '.npy', echo_noise_list)
    echo_noise_list = []

    for root, dirs, files in os.walk(T2_AMPLITUDE_path):
        for file in files:
            T2_AMPLITUDE = np.load(T2_AMPLITUDE_path + file)
            T2_AMPLITUDE_list.append(list(T2_AMPLITUDE))
    T2_AMPLITUDE_list = np.array(T2_AMPLITUDE_list).reshape(-1, 128)
    np.save(save_T2_AMPLITUDE_path + 'T2_AMPLITUDE'+'_SNR' + str(SNR) + '.npy', T2_AMPLITUDE_list)
    T2_AMPLITUDE = []

    return "Merge is finish"



def get_matrix_a(echo_num, echo_space, t2_sample_num, t2_min, t2_max):
    t_list = []
    # 用秒的单位
    for i in range(echo_num):
        t_list.append((i + 1) * echo_space)
    t = np.array(t_list)
    t = np.expand_dims(t, axis=-1)
    log_min, log_max = math.log10(t2_min), math.log10(t2_max)
    t2 = np.logspace(log_min, log_max, t2_sample_num, base=10)
    t2 = np.expand_dims(t2, axis=0)
    t2 = 1 / t2
    matrix_a = np.dot(t, t2)
    matrix_a = np.exp(-1 * matrix_a)
    return matrix_a


def normal_distribution(x, mean, sigma):
    """
    :param x: x
    :param mean: 正态分布的均值
    :param sigma:标准差
    :return: 正态分布的T2幅度值
    """
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))


def forward_T2(ratio, T2_number, position, sigma, SNR):
    # 回波间隔
    TE = 0.2
    # 回波个数
    NE = 2501
    # 孔隙度 [0.1, 2.4, 50]
    por = 1
    # 高粘度有机质
    por_1 = por * ratio[0]
    # 束缚水
    por_2 = por * ratio[1]
    # T2布点时间最小值
    T2_min = 0.01
    # T2布点时间最大值
    T2_max = 10000
    # T2布点
    T2 = np.logspace(np.log10(T2_min), np.log10(T2_max), T2_number)
    x = np.linspace(0, 1, T2_number)

    position_1 = (math.log10(position[0])-math.log10(0.01))/(math.log10(10000)-math.log10(0.01))
    position_2 = (math.log10(position[1])-math.log10(0.01))/(math.log10(10000)-math.log10(0.01))

    Model_1 = normal_distribution(x, position_1, sigma[0])
    Model_2 = normal_distribution(x, position_2, sigma[1])

    f1 = Model_1 * por_1 / sum(Model_1)
    f2 = Model_2 * por_2 / sum(Model_2)

    # 验证孔隙度是否为10
    f = f1 + f2
    # sum_f = sum(f)
    f = f.reshape((T2_number, 1))
    # 此处需要考虑精度问题
    A = get_matrix_a(NE, TE, T2_number, T2_min, T2_max)
    x = np.dot(A, f)
    deta = x[1] / SNR
    noise = np.random.normal(loc=0, scale=1, size=(NE, 1)) * deta
    # print(noise)
    y = x + noise
    f = f.reshape((1, T2_number))
    x_return = x[1:2501]
    y_return = y[1:2501]
    return x_return, y_return, f

def data_generated(T2_number, SNR, pure_echo_path, echo_noise_path, T2_AMPLITUDE_path):
    T2_min = 0.01
    # T2布点时间最大值
    T2_max = 10000
    # 无噪声回波串
    pure_echo = []
    # 有噪声回波串
    echo_noise = []
    # 幅度
    T2_AMPLITUDE = []
    sigma_list = np.linspace(0.05, 0.10, 5)
    # position_list = np.logspace(np.log10(1), np.log10(150), 70)
    T2 = np.logspace(math.log10(T2_min), math.log10(T2_max), T2_number)
    position_list = T2[42: 90]
    # position2_list = np.logspace(np.log10(1), np.log10(150), 50)
    ratio_list = np.linspace(0, 1, 10)
    print(position_list)

    for ratio_num in range(ratio_list.shape[0]):
        ratio1 = ratio_list[ratio_num]
        ratio2 = 1 - ratio1
        ratio = [ratio1, ratio2]
        for sigma_num1 in range(sigma_list.shape[0]):
            # sigma = sigma_list[sigma_num1]
            for sigma_num2 in range(sigma_list.shape[0]):
                for num1 in range(position_list.shape[0]):
                    for num2 in range(num1, position_list.shape[0]):
                        position = [position_list[num1], position_list[num2]]
                        sigma = [sigma_list[sigma_num1], sigma_list[sigma_num2]]
                        x, y, f = forward_T2(ratio, T2_number, position, sigma, SNR)
                        pure_echo.append(list(x))
                        echo_noise.append(list(y))
                        T2_AMPLITUDE.append(list(f))
                        print("ratio is " + str(ratio) +  "; sigma is " + str(sigma) + "; position is " + str(position))
        pure_echo = np.array(pure_echo).reshape(-1, 2500)
        echo_noise = np.array(echo_noise).reshape(-1, 2500)
        T2_AMPLITUDE = np.array(T2_AMPLITUDE).reshape(-1, 128)
        np.save(pure_echo_path + 'pure_echo_' + str(ratio_num) + str(sigma_num1)+ '.npy',
                pure_echo)
        np.save(echo_noise_path + 'echo_noise_' + str(ratio_num) + str(sigma_num1) + '.npy',
                echo_noise)
        np.save(T2_AMPLITUDE_path + 'T2_AMPLITUDE_' + str(ratio_num) + str(sigma_num1) + '.npy',
                T2_AMPLITUDE)
        pure_echo = []
        echo_noise = []
        T2_AMPLITUDE = []

