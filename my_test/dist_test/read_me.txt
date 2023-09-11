python -m torch.distributed.launch --nproc_per_node=2 original.py 
#to generate x.pth, weights.pth, bias.pth, output.pth.

 python -m torch.distributed.launch --nproc_per_node=2 dist_column_linear_copy.py 
 #get the x.pth , weights, bias, to generate output, compare with loaded_output from output.pth, they are equal.