# import torch
# import torch.nn as nn

# # Create a tensor of shape (4, 256, 768)
# input_tensor = torch.randn(4, 256, 768)

# # Initialize LayerNorm
# layer_norm = nn.LayerNorm(input_tensor.shape[2:])  # Apply LayerNorm to the last dimension (768)

# # Apply LayerNorm individually to each dimension
# output_individually = torch.stack([layer_norm(input_tensor[:, :, i]) for i in range(input_tensor.shape[2])], dim=2)

# # Check if the results are close
# if torch.allclose(output_individually, output_individually):
#     print("LayerNorm applied individually to each dimension.")
# else:
#     print("LayerNorm applied individually to each dimension produces different results.")

import torch

# Create a tensor of shape (4, 256, 768)
input_tensor = torch.randn(4, 256, 768)

# Specify the dimension you want to apply LayerNorm to
dimension = 2  # Apply LayerNorm to the last dimension (768 in this case)

# Calculate mean and variance along the specified dimension
mean = input_tensor.mean(dim=dimension, keepdim=True)
variance = input_tensor.var(dim=dimension, keepdim=True)

# Apply LayerNorm manually
normalized_tensor = (input_tensor - mean) / torch.sqrt(variance + 1e-12)

# You can add scaling and shifting if needed
# normalized_tensor = normalized_tensor * scale + shift

# Check the shape of the normalized tensor
print(normalized_tensor.shape)
print(normalized_tensor[:,0:1,:])

first_tensor = input_tensor[:,0:1,:]
mean_1st = first_tensor.mean(dim=dimension, keepdim=True)
variance_1st = first_tensor.var(dim=dimension, keepdim=True)

# Apply LayerNorm manually
normalized_tensor_1st = (first_tensor - mean_1st) / torch.sqrt(variance_1st + 1e-12)
print(normalized_tensor_1st)

