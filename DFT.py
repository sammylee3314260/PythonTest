def dft(x): # x is a list with length N (time domain)
    import math
    N = len(x)
    X = [0]*N
    for k in range(N):
        X[k] = 0
        for n in range(N):
            angle = -2*math.pi*k*n/N
            X[k]+= x[n] * complex(math.cos(angle), -1*math.sin(angle))
    return X

def idft(X): #X is a list with length N (frequency domain)
    N = len(X)
    x = [0]*N
    for n in range(N):
        x[n] = 0
        for k in range(N):
            angle = 2*math.pi*k*n/N
            x[n] += X[k] * complex(math.cos(angle),math.sin(angle))
        x[n] = x[n] / N
    return x
'''
Function DFT(x):
    Input: x, a list of N real or complex numbers
    Output: X, a list of N complex numbers representing frequency domain

    N = length(x)
    X = list of length N (initialize with 0s)

    For k from 0 to N-1:
        X[k] = 0
        For n from 0 to N-1:
            angle = -2 * PI * k * n / N
            X[k] += x[n] * (cos(angle) - i * sin(angle))

    Return X

Function IDFT(X):
    Input: X, a list of N complex numbers (frequency domain)
    Output: x, a list of N complex numbers (time domain)

    N = length(X)
    x = list of length N (initialize with 0s)

    For n from 0 to N-1:
        x[n] = 0
        For k from 0 to N-1:
            angle = 2 * PI * k * n / N
            x[n] += X[k] * (cos(angle) + i * sin(angle))
        x[n] = x[n] / N

    Return x
'''

