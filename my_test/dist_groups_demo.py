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

        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(split_size, input_size)).to(world_rank) 
            for split_size in split_sizes
        ])
        self.bias = nn.Parameter(torch.randn(output_size)).to(world_rank)

    def forward(self, x):
        results = []
        for i, weight in enumerate(self.weights):
            with torch.cuda.device(i):
                xi = x.to(i)
                yi = torch.matmul(xi, weight.t()).to(world_rank)
                results.append(yi)

        y = torch.cat(results, dim=1) + self.bias
        return y

# Example usage:
model = DynamicColumnParallelLinear(1000, 500)
x = torch.randn(32, 1000).to(world_rank)
output = model(x)
print(output.shape) # Should be torch.Size([32, 500])

# Cleanup
dist.destroy_process_group()
