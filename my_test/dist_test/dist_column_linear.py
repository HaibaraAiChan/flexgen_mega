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
        loaded_weights = torch.load('weights.pth')
        self.weights = nn.ParameterList()
        start = 0
        for split_size in split_sizes:
            end = start + split_size
            self.weights.append(nn.Parameter(loaded_weights[start:end].to(world_rank)))
            start = end
            
        loaded_bias = torch.load('bias.pth')
        self.bias = nn.ParameterList()
        start = 0
        for split_size in split_sizes:
            end = start + split_size
            self.bias.append(nn.Parameter(loaded_bias[start:end].to(world_rank)))
            start = end
       
        print('output size ', output_size)
        print('self.num_gpus', self.num_gpus)
        # output_per_device = int(output_size/ self.num_gpus)
        

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
# x = torch.randn(32, 1000).to(world_rank)
# Load x from the file
loaded_x = torch.load('x.pth')
x = loaded_x.to(world_rank)

output = model(x)
print()
print('device of output ', output.device)
print(output.shape)  # Should be torch.Size([32, 500])
# Load output from the original output file
loaded_output = torch.load('output.pth').to(output.device)

# Cleanup
dist.destroy_process_group()

are_equal = torch.equal(output, loaded_output)
print('equal ', are_equal)