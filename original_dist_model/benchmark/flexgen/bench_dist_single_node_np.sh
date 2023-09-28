#!/bin/bash

MY_IPADDR=$(hostname -i)
all_hosts=$MY_IPADDR
N_GPUS=4
N_CORES_PER_GPU=4

# PYTHON_EXEC=$CONDA_PREFIX/bin/python
PYTHON_EXEC=home/cc/LLM/bin/python
PYTHON_SCRIPT=flexgen.dist_flex_opt
# PYTHON_SCRIPT=home/cc/Flexgen/flexgen/flex_opt.py
pgrep -fl python | awk '!/dist_flex_opt\.py/{print $1}' | xargs sudo kill 
# finds and kills certain Python processes

set -x  # activates debugging mode, printing each subsequent command before it is executed.
echo "after set -x"
mpirun --verbose \
  --mca btl sm,self \
  --map-by ppr:2:node:pe=4 \
  --bind-to core -x OMP_NUM_THREADS=4 \
  python -m flexgen.dist_flex_opt \
    --head-ip 127.0.1.1 \
    --port 7777 \
    --use-mpi \
    --model facebook/opt-6.7b \
    --gpu-batch-size 8 \
    --percent 0 100 0 100 0 100  \
    --comm-device cpu \
    --path _DUMMY_ \
    --cut-gen-len 5 \
    --cpu

