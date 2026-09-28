#https://gemini.google.com/share/56aa2cd70669


from mpi4py import MPI
from math import sqrt
def compute_term(k):
    s = 0
    for i in range(k//1000):
        s+= i*i
    return 1.0/(k*k)

N=500000
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Hello from process {rank} out of {size} processes")

s = 0

start_time = MPI.Wtime()

for i in range(1+rank, N+1, size):
    s+= compute_term(i)

end_time = MPI.Wtime()
duration = end_time - start_time
print(f"Process {rank} computed its partial sum in {duration:.4f} seconds.")

total_s = comm.reduce(s, op=MPI.SUM, root=0)

if rank == 0:
    pi = sqrt(6.0 * total_s)
    print(f"Calculation of pi: {pi}")

