import numpy as np
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt
import math


def adjust_data(x, y, debug=0):

    # 開始点
    index0 = np.argmax(y)
    # 終点
    index1 = int(y.size / 2)
    temp1 = np.argmax(y[index1:])
    index1 += (temp1 + 1)

    x1 = x[index0:index1]
    y1 = y[index0:index1]

    if debug == 1:
        plt.plot(x[index0], y[index0], "ro")
        plt.plot(x[index1], y[index1], "ro")
        plt.plot(x, y, "g", label="before")
        plt.plot(x1, y1, "b", label="before")
        plt.legend()
        plt.show()

    return x1, y1


def get_key(x, y, debug=0):

    # データの補正
    x, y = adjust_data(x, y, debug)

    # FFT
    freq = fftpack.fftfreq(y.size, d=50)
    F = fftpack.fft(y)
    pidxs = np.where(freq > 0)
    freqs, power = freq[pidxs], np.abs(F)[pidxs]

    if debug == 1:
        # フーリエ変換の結果
        plt.plot(freqs, power)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.show()

    # フィルター
    n = 0.003
    # n = 0.005
    F[np.abs(freq) > n] = 0
    # n = 0.0008
    # n = 0.002
    # F[np.abs(freq) < n] = 0

    # IFFT
    G = np.real(fftpack.ifft(F))

    # ピークの検出
    # 極大値
    # maximal_idx = signal.argrelmax(G, order=42)
    # 極小値
    minimal_idx = signal.argrelmin(G, order=30)

    # プロット
    if debug == 1:
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.plot(x, y, "g", label="before")
        ax1.set_title("FFT before")
        ax2.plot(x, G, label="after")
        ax2.set_title("FFT after")

        fig.tight_layout()
        plt.show()

        # plt.plot(x[maximal_idx],G[maximal_idx],'ro',label='peak_maximal') # 極大点プロット
        # plt.plot(x[maximal_idx],G[maximal_idx],'bo',label='peak_maximal') # 極大点プロット
        plt.plot(x[minimal_idx], G[minimal_idx], 'bo',
                 label='peak_maximal')  # 極小点プロット
        plt.plot(x, G, label="after")
        plt.show()

    # 計算
    temp = x[minimal_idx]
    com = []
    e = []

    for i in range(temp.size - 1):
        com.append(temp[i + 1] - temp[i])
        e.append(np.abs(np.log10(temp[i + 1] - temp[i])))

    ave = math.ceil(np.average(e)) + 1
    for i in range(len(com)):
        print(com[i] * (10**ave))
        com[i] = math.ceil(com[i] * (10**ave))
    print(com)
    temp2 = sorted(list(set(com)))
    juge = np.average(temp2)

    result = "0b1"
    for i in range(len(com)):
        if com[i] >= juge:
            result += "1"
        else:
            result += "0"

    # 結果の出力
    print(result)
    print(hex(int(result[2:], 2)))
