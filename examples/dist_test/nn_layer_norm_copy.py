import torch
import torch.nn.functional as F
import torch.nn as nn

# Sample input tensor of shape (4, 256, 768)
# input_tensor = torch.randn(4, 6, 8)

input_tensor1 = torch.tensor([[[ 0.7140,  1.2694,  1.9161, -0.3093, -0.4700, -0.0953, -0.3145,
          -0.3397],
         [-0.1076, -0.3316, -0.2506, -1.3907,  0.0831, -0.1975,  2.2081,
          -1.0653],
         [ 0.1837,  0.3289, -0.6912,  0.9001,  0.2444, -1.3712, -0.1864,
          -0.8603],
         [-2.3834, -0.2209, -0.1459, -0.7660, -1.2501,  1.6441, -0.5666,
           2.8049],
         [ 1.8633, -2.4605, -0.2591,  1.6334, -0.4565, -1.5432,  0.4368,
          -0.6201],
         [ 0.4783,  0.1436,  0.4203, -1.4026,  1.3363, -0.6369,  0.9614,
           0.1896]],

        [[-0.7618, -1.2970, -0.8145, -0.1886,  1.3093, -1.4779, -0.5507,
          -0.1643],
         [ 0.8997,  0.4507, -1.0494,  0.3268,  1.2872,  0.0682, -0.5764,
           0.5266],
         [ 0.2426, -0.5346, -0.9074, -1.6059, -0.6010, -0.7947,  0.7443,
           0.5497],
         [-1.0009,  1.8454,  0.2033, -1.3620,  0.8349,  0.9912, -0.6945,
           1.2158],
         [ 0.8007,  0.3546,  1.1182, -0.3110,  0.4730, -0.3037, -1.6054,
           1.1016],
         [ 2.0499,  2.0012,  0.2598,  1.5545, -0.6951,  0.8456, -0.6067,
          -1.6209]],

        [[-0.0690,  2.2011,  0.1708, -0.1260, -0.4826, -0.6559, -1.1875,
          -0.4103],
         [-0.1574, -0.1909,  0.3432,  1.0440, -0.0073, -1.0900, -0.9364,
           0.1869],
         [-0.8465,  0.1999,  0.4742, -0.0994, -1.5376,  1.5575,  0.7529,
          -0.6666],
         [-0.9510,  1.4929, -0.2562,  0.6281, -2.0647, -0.4667, -0.3943,
          -0.4406],
         [-0.3665, -2.2512, -0.7605,  0.7126, -0.2672, -0.8038, -0.1394,
           0.2349],
         [-1.2042, -0.2385,  0.8350, -0.3792, -0.1433, -0.4046,  0.3056,
          -0.6243]],

        [[ 0.1698, -0.7681,  0.3594,  2.0709,  0.0523,  1.4570, -1.1011,
           1.3996],
         [ 1.3160, -0.3206, -0.9684, -2.2957,  0.0897,  0.2080,  1.8940,
           2.2434],
         [ 2.1560, -0.0254,  0.8072,  1.3866,  1.3058, -0.2193,  0.4203,
           2.7514],
         [-1.6150,  0.2054,  0.5374,  2.1578, -1.4743,  0.6965,  2.6361,
          -0.3889],
         [-0.2644,  0.4464,  0.7153, -0.2296,  0.2220, -2.5365, -0.5849,
          -0.3764],
         [-0.7026, -1.6506, -1.1739,  0.6246, -0.0983,  0.1501, -0.8499,
           0.1895]]])
print('input_tensor ', input_tensor1.shape)
input_tensor1 = input_tensor1.permute(1,0,2)
print('input_tensor permute ', input_tensor1.shape)

# Applying LayerNorm across dimension 2 (the entire tensor)
print('input_tensor1[2].shape ', input_tensor1[2].shape)
output_whole = F.layer_norm(input_tensor1, input_tensor1[2].shape)
# print('output_whole ', output_whole[:,0:1,:])

# Applying LayerNorm individually to each (4, 1, 768) slice along dimension 2
slices = torch.chunk(input_tensor1, input_tensor1.size(0), dim=0)
print('slices[0].shape ', slices[0].shape)

layer_norm_slices = [F.layer_norm(slice, input_tensor1[2].shape) for slice in slices]
output_slices = torch.cat([layer_norm for layer_norm in layer_norm_slices], dim=0)

# print('output_slices ', output_slices[:,0:1,:])
# Check if the results are the same
output_whole= output_whole.permute(1,0,2)
output_slices= output_slices.permute(1,0,2)
if torch.allclose(output_whole, output_slices, atol=1e-6):
    print("LayerNorm results are the same.")
else:
    print("LayerNorm results are different.")

# Print the shapes of the outputs
print("Shape of output_whole:", output_whole.shape)  # Should be (4, 256, 768)
print("Shape of output_slices:", output_slices.shape)  # Should be (4, 256, 768)
