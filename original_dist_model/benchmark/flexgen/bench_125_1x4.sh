#!/bin/bash

MY_IPADDR=$(hostname -i)
all_hosts=$MY_IPADDR
N_GPUS=2
N_CORES_PER_GPU=6

PYTHON_EXEC=python
PYTHON_SCRIPT=flexgen.dist_flex_opt

pgrep -fl python | awk '!/dist_flex_opt\.py/{print $1}' | xargs sudo kill

set -x

mpirun \
  --mca btl sm,self\
  --map-by ppr:$N_GPUS:node:pe=$N_CORES_PER_GPU \
  --bind-to core -x OMP_NUM_THREADS=$N_CORES_PER_GPU \
  $PYTHON_EXEC -m $PYTHON_SCRIPT \
    --head-ip $MY_IPADDR \
    --port 7777 \
    --use-mpi \
    --model facebook/opt-125m \
    --gpu-batch-size 12 \
    --percent 100 0 100 0 100 0\
    --comm-device gpu \
    --cut-gen-len 5 \
    --path _DUMMY_ \
    --cpu
