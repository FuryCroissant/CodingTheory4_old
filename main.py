import bitarray as ba
import numpy as np
import math
class GolayCode:
    def __init__(self):
        self.k = int(12)
        self.n = int(24)
        self.B = np.array([[1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                           [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
                           [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                           [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                           [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
                           [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                           [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
                           [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                           [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                           [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
                           [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

        self.I = np.mat(np.eye(12), dtype = int) #единичная матрица 12х12

        self.H = np.concatenate((self.I, self.B))#проверочная матрица H

        self.G = np.transpose(self.H)#матрица расширенного кода Голея G

    def Encode(self, input):
        return np.dot(input, self.G) % 2

    def WH(self, a):  # расчет веса Хэмминга
        wt = 0
        for i in range(int(len(a))):
            if a[i] == 1:
                wt += 1
        return wt

    def Decode(self, input):
        u = [] 
        zero = [0]*12
        e = [0]*12
        synd = np.dot(input, self.H) % 2
        ss1 = []
        for i in range(0, 12):
            ss1.append(int(synd[0, i]))
        if self.WH(ss1) <= 3:
            u = np.concatenate((ss1, zero))
        for j in range(0, 12):
            ss2 = (ss1+self.B[j]) % 2
            if self.WH(ss2) <= 2:
                e[j] = 1
                u = np.concatenate((ss2, e))
        synd2 = np.dot(synd, self.B)% 2
        S3 =[]
        for i in range(0, 12):
            S3.append(int(synd2[0, i]))
        if self.WH(S3) <= 0:
            u = np.concatenate((zero, S3))
        for k in range(0, 12):
            ss4 = (S3+self.B[k]) % 2
            if self.WH(ss4) <= 2:
                e[k] = 1
                u = np.concatenate((e, ss4))
        v=(input+u)%2
        return v

class RMCode:

    def __init__(self):
        self.r = int(1)
        self.m = int(3)
        self.k = int(0)
        for i in range(self.r + 1):
            self.k = self.k + int(math.factorial(self.m) / (math.factorial(i) * math.factorial(self.m - i)))
        self.n = 2 ** self.m

    def g(self, r, m):
        if (r == 0):
            return np.ones(2 ** m, dtype=int)
        if (r == m):
            s = np.zeros(2 ** m, dtype=int)
            s[len(s) - 1] = 1
            result = self.g(m - 1, m)
            return np.vstack((result, s))
        cur1 = np.concatenate((self.g(r, m - 1), self.g(r, m - 1)), axis=1)
        cur2 = self.g(r - 1, m - 1)
        if (len(cur2.shape) == 1):
            z = np.zeros(len(cur1[0]) - len(cur2), dtype=int)
            cur2 = np.append(z, cur2)
        else:
            z = np.zeros((len(cur2), (len(cur1[0]) - len(cur2[0]))), dtype=int)
            cur2 = np.concatenate((z, cur2), axis=1)
        return np.vstack((cur1, cur2))

    def Encode(self, a):
        G = self.g(self.r, self.m)
        return np.dot(a, G) % 2

    def h(self, i, m):
        I1 = np.eye(2 ** (m - i), dtype=int)
        I2 = np.eye(2 ** (i - 1), dtype=int)
        H = np.array([[1, 1], [1, -1]], dtype=int)
        for i in range(len(I1)):
            H_n = I1[i][0] * H
            for j in range(1, len(I1)):
                H_n = np.concatenate((H_n, I1[i][j] * H), axis=1)
            if (i == 0):
                H2 = H_n
            else:
                H2 = np.concatenate((H2, H_n))
        for i in range(len(H2)):
            H_n = H2[i][0] * I2
            for j in range(1, len(H2)):
                H_n = np.concatenate((H_n, H2[i][j] * I2), axis=1)
            if (i == 0):
                result = H_n
            else:
                result = np.concatenate((result, H_n))
        return result

    def Decode(self, w):
        w_n = w.copy()
        for i in range(len(w)):
            if (w_n[i] == 0):
                w_n[i] = -1
        w1 = np.dot(w_n, self.h(1, self.m))
        wi = np.array([w1, np.dot(w1, self.h(2, self.m))])
        for i in range(2, self.m):
            wi = np.vstack((wi, np.dot(wi[i - 1], self.h(i + 1, self.m))))
        max_value = max(wi[len(wi) - 1], key=abs)
        for i in range(len(wi[0])):
            if (wi[len(wi) - 1][i] == max_value):
                break
        j = bin(i)
        j = [int(x) for x in list(j[2:len(j)])]
        if (len(j) < self.n - self.k - 1):
            j = np.append(np.zeros(self.n - self.k - 1 - len(j), np.int), j)
        j = np.flipud(j)
        if (max_value > 0):
            j = np.append(1, j)
        else:
            j = np.append(0, j)
        return j

    def Read(self, name):
        path = "C:\\Users\\Маша\\PycharmProjects\\TK4\\" + name
        f = open(path, 'rb')
        v = ba.bitarray()
        v.fromfile(f)
        v = list(v.to01())
        return [int(x) for x in v]

    def Write(self, name, v):
        path = "C:\\Users\\Маша\\PycharmProjects\\TK4\\" + name
        f = open(path, 'wb')
        v.tofile(f)

    def EncodeFile(self, name):
        v = self.Read(name)
        result = np.zeros(0, np.int)
        for i in range(int(len(v) / self.k)):
            j = i * self.k
            c = v[j: j + self.k]
            c = self.Encode(c)
            result = np.append(result, c)
        return result

    def DecodeFile(self, name, f):
        result = np.zeros(0, np.int)
        for i in range(int(len(f) / self.n)):
            j = i * self.n
            c = f[j: j + self.n]
            result = np.append(result, self.Decode(c))
        self.Write(name, np.packbits(result))




GC = GolayCode()
a = np.array([1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1])
p1 = np.array([1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
p2 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0])
A = GC.Encode(a)
B = GC.Decode(p1)
C = GC.Decode(p2)
print("Часть 4.1")
print("№4.1.2\nВходной сигнал: ", a, "\nВыходной сигнал:", A)
print("№4.1.3\nПример 1\nВходной сигнал : ", p1, "\nВыходной сигнал: ", B, "\nПример 2\nВходной сигнал : ", p2, "\nВыходной сигнал: ", C)

rmc = RMCode()
G= rmc.g(1, 3)
d = np.array([1, 0, 1, 1])
c = np.array([1, 0, 0, 0, 1, 1, 1, 1])
E = rmc.Encode(d)
F = rmc.Decode(E)
v = rmc.EncodeFile("doc.txt")
rmc.DecodeFile("out1.txt", v)
v[3] = not v[3]
rmc.DecodeFile("out2.txt", v)
print("\nЧасть 4.2")
print("№4.2.2\nМатрица G(1,3):\n", G, "\nИсходная последовательность:", d,"\nЗакодированная последовательность:", E)
print("№4.2.3\nДекодированная последовательность:", F)
print("№4.2.4\nИсходный файл: ", rmc.Read("doc.txt"),"\nДекодированный файл без ошибки:", rmc.Read("out1.txt"),"\nДекодированный файл с ошибкой :", rmc.Read("out2.txt"))