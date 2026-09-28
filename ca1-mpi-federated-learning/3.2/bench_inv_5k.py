#https://gemini.google.com/share/7814678e68c1
#my prompts:
# the function to create random matrix 
# funciton to compute its inversion using numpy

import numpy as np
import time
N = 5000

matrix = np.random.rand(N, N)

start_time = time.perf_counter()
inv_matrix = np.linalg.inv(matrix)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Time taken to compute the inversion of a {N}x{N} matrix: {elapsed_time:.2f} seconds")

print("Inversion complete.")

print(inv_matrix[N//2, N//2])