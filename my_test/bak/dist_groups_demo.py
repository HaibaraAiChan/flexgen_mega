# import torch
# import torch.nn as nn
# import torch.distributed as dist

# dist.init_process_group(backend='gloo')
# world_size = dist.get_world_size()
# world_rank = dist.get_rank()

# class DynamicColumnParallelLinear(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(DynamicColumnParallelLinear, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size

#         self.num_gpus = world_size
#         split_sizes = [output_size // self.num_gpus for _ in range(self.num_gpus)]
#         split_sizes[-1] += output_size % self.num_gpus

#         self.weights = nn.ParameterList([
#             nn.Parameter(torch.randn(split_size, input_size).to(world_rank))
#             for split_size in split_sizes
#         ])
#         print('world rank ', world_rank)
#         print('weight size ', len(self.weights))
#         print('weight size ', self.weights[0].size())
#         self.bias = nn.Parameter(torch.randn(output_size).to(world_rank))

#     def forward(self, x):
#         results = []
#         for i, weight in enumerate(self.weights):
#             xi = x.to(world_rank)  # Move x to the same device as weight
#             yi = torch.matmul(xi, weight.t())
#             results.append(yi)

#         y = torch.cat(results, dim=1) + self.bias
#         return y

# # Example usage:
# model = DynamicColumnParallelLinear(1000, 500)
# x = torch.randn(32, 1000).to(world_rank)
# output = model(x)
# print('device of output ', output.device)
# print(output.shape)  # Should be torch.Size([32, 500])

# # Cleanup
# dist.destroy_process_group()


import torch
import torch.nn as nn
import torch.distributed as dist

dist.init_process_group(backend='gloo')
world_size = dist.get_world_size()
world_rank = dist.get_rank()

class DynamicColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(DynamicColumnParallelLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.num_gpus = world_size
        split_sizes = [output_size // self.num_gpus for _ in range(self.num_gpus)]
        split_sizes[-1] += output_size % self.num_gpus
        print('split sizes ', split_sizes)
        
        # Load weights and bias from files
        # loaded_weights = torch.load('weights.pth')
        # loaded_bias = torch.load('bias.pth')
        
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(split_size, input_size).to(world_rank))
            for split_size in split_sizes
        ])
        print('output size ', output_size)
        print('self.num_gpus', self.num_gpus)
        # output_per_device = int(output_size/ self.num_gpus)
        self.bias = nn.ParameterList([
            nn.Parameter(torch.randn(int(split_size/sum(split_sizes)*output_size)).to(world_rank))
            for split_size in split_sizes
        ])

    def forward(self, x):
        results = []
        for i, (weight, bias) in enumerate(zip(self.weights, self.bias)):
            xi = x.to(world_rank)  # Move x to the same device as weight
            yi = torch.matmul(xi, weight.t())  # Perform linear transformation
            print('rank ', i)
            print('y size ', yi.size())
            yi = yi + bias
            results.append(yi)

        y = torch.cat(results, dim=1)   # Concatenate and add bias
        return y

# Example usage:
model = DynamicColumnParallelLinear(1000, 500)
x = torch.randn(32, 1000).to(world_rank)
output = model(x)
print()
print('device of output ', output.device)
print(output.shape)  # Should be torch.Size([32, 500])

# Cleanup
dist.destroy_process_group()
