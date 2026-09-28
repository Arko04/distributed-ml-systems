#https://gemini.google.com/share/1562ca5af007
#my prompts:
# the function to create random matrix 
# funciton to compute its determinant using numpy

import numpy as np
import time

N = 8000
M = 8000

matrix = np.random.rand(N, M)

start_time = time.perf_counter()
sign, logdet = np.linalg.slogdet(matrix)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Time taken to compute the determinant of a {N}x{M} matrix: {elapsed_time:.2f} seconds")

print(f"\nThe determinant is (theoretically): {sign} * e^({logdet:.2f})")