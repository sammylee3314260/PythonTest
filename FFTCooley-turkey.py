def fft(x):
    import math
    N = len(x)
    if N == 1:  return x # base case
    # divide
    evenx = []
    oddx = []
    for i in range(N):
        if i%2: evenx.append(x[i])
        else: oddx.append(x[i]
    even = FFT(evenx)
    odd = FFT(oddx)
    #merge
    X = [0]*N
    for k in range(N/2-1):
        twiddle = math.exp(-2*math.pi*k/N)
        X[k] = even[k]+twiddle*odd[k]
        X[k+N/2] = even[k]-twiddle*odd[k]
    return X
'''
function FFT(x):
    N = length(x)
    if N == 1:
        return x  // base case

    // Step 1: 分成偶數與奇數子序列
    even = FFT(x[0], x[2], ..., x[N-2])
    odd  = FFT(x[1], x[3], ..., x[N-1])

    // Step 2: 初始化空陣列
    X = array of size N

    for k = 0 to N/2 - 1:
        twiddle = exp(-2πi * k / N)
        X[k]         = even[k] + twiddle * odd[k]
        X[k + N/2]   = even[k] - twiddle * odd[k]

    return X
'''
