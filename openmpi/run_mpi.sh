#!/bin/bash

# Set the number of processes (adjust as needed)
NUM_PROCESSES=4

# The name of your Python MPI script
MPI_SCRIPT="./my_mpi_program.py"

# Optional: Any additional MPI options you want to use
MPI_OPTIONS="-np $NUM_PROCESSES"

# Run the Python MPI script
mpirun $MPI_OPTIONS python $MPI_SCRIPT

# Add any additional post-processing commands here
