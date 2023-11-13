import torch

# Create a tensor (example)
original_tensor = torch.tensor([1.23456789, 2.3456789, 3.456789])

# Format the tensor items with four decimal places using vectorized operations
formatted_tensor = original_tensor.round(.4)

# Print the formatted tensor
print(formatted_tensor)
