import numpy as np


def method_kramer(A, b):
    count = len(A)
    delta = np.linalg.det(A)
    x = []
    for j in range(count):
        buf = np.copy(A)
        buf[:, j] = b
        x.append(float("%.5f" % (np.linalg.det(buf)/delta)))
    return x


def method_jacobi (a, b, eps):
    count = len(a)
    x = [0, 0, 0, 0]
    temp = [0, 0, 0, 0]
    norm = 1
    while norm > eps:
        for i in range(count):
            temp[i] = b[i]
            for g in range(count):
                if i != g:
                    temp[i] -= a[i][g]*x[g]
            temp[i] /= a[i][i]
        norm = abs(x[0]-temp[0])
        for k in range(count):
            if abs(x[k] - temp[k]) > norm:
                norm = abs(x[k] - temp[k])
            x[k] = temp[k]
    for k in range(count):
        x[k] = (float("%.5f" % (x[k])))
    return x


def matrix_max_row(matrix, n):
    max_elem = matrix[n][n]
    max_row = n
    for i in range(n + 1, len(matrix)):
        if abs(matrix[n][i]) > abs(max_elem):
            max_elem = matrix[n][i]
            max_row = i
        if max_row != n:
            matrix[n], matrix[max_row] = matrix[max_row], matrix[n]


def method_gauss(a):
    n = len(a)
    x = np.zeros(n)
    for k in range(n - 1):
        matrix_max_row(a, k)
        for i in range(k + 1, n):
            div = a[i][k] / a[k][k]
            a[i][-1] -= div * a[k][-1]
            for j in range(k, n):
                a[i][j] -= div * a[k][j]
    if is_singular(a):
        print('The system has infinite number of answers')
        return
    for k in range(n - 1, -1, -1):
        x[k] = (a[k][-1] - sum([a[k][j] * x[j] for j in range(k + 1, n)])) / a[k][k]
        x[k] = float("%.5f" % x[k])
    return x


def is_singular(matrix):
    for i in range(len(matrix)):
        if not matrix[i][i]:
            return True
        return False
a = [[25.0, -1.0, 1.0, 4.0],
     [1.0, 23.0, -7.0, 1.0],
     [1.0, 2.0, 10.0, 2.0],
     [12.0, 56.0, 1.0, 30.0]]
b = [41, -19, 34, 39]
c = [[25.0, -1.0, 1.0, 4.0, 41],
     [1.0, 23.0, -7.0, 1.0, -19],
     [1.0, 2.0, 10.0, 2.0, 34],
     [12.0, 56.0, 1.0, 30.0, 39]]
eps = 0.00001
X1 = np.linalg.solve(a, b)
print('With np.linalg.solve function:')
print('X = ', X1)
X2 = method_kramer(a, b)
print('Kramer method:')
print('X = ', X2)
X3 = method_jacobi(a, b, 0.00001)
print('Jacobi method:')
print('X = ', X3)
X4 = method_gauss(c)
print('Gauss method:')
print('X = ', X4)
