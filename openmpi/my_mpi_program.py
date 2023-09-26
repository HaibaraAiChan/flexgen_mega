# my_mpi_program.py

from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print('rank ', rank)
size = comm.Get_size()
print('the total number of processes  ',size) # the total number of processes in that communicator.

# Define a list of numbers to sum (modify as needed)
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Calculate the local sum for each process
local_sum = sum(numbers[rank::size])
print('numbers[rank::size] ', numbers[rank::size])
print('local sum ', local_sum)
# Gather all local sums to the root process (rank 0)
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

# Print the results
if rank == 0:
    print(f"Total sum: {total_sum}")

# Finalize MPI
MPI.Finalize()
