import torch
tensor_parallel_size = 2
b = 4 # batch size
n_head = 12
step = n_head // tensor_parallel_size
original_tensor = torch.arange(48)

indices=[]
for bb in range(0,b*2,2):
    indices.append(list(range(bb*step,(bb+1)*step)))
    
# Define the indices for splitting
# indices = [list(range(6)), list(range(12, 18)), list(range(24, 30)), list(range(36, 42))]
print('indices ', indices)
# Convert indices to a flat tensor
flat_indices = torch.tensor([idx for sublist in indices for idx in sublist])
print('flat_indices ', flat_indices)
# Use index_select to extract the desired elements
sub_tensors = original_tensor[flat_indices]
print(sub_tensors)