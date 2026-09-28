from math import sqrt
import time

def compute_term(k):
    s = 0
    for i in range(k//1000):
        s+= i*i
    return 1.0/(k*k)

def compute_Basel(N):
    s = 0
    for i in range(1,N+1):
        s+= compute_term(i)
    return sqrt(6.0*s)
    
pi = compute_Basel(500000)

print(f"Calculation of pi: {pi}")