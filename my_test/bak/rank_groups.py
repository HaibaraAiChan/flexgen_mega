import torch
import torch.nn as nn

class DynamicColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(DynamicColumnParallelLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.num_gpus = torch.cuda.device_count()
        split_sizes = [output_size // self.num_gpus for _ in range(self.num_gpus)]
        split_sizes[-1] += output_size % self.num_gpus

        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(split_size, input_size)).cuda(i) 
            for i, split_size in enumerate(split_sizes)
        ])
        self.bias = nn.Parameter(torch.randn(output_size)).cuda(0)

    def forward(self, x):
        results = []
        for i, weight in enumerate(self.weights):
            xi = x.cuda(i)
            yi = torch.matmul(xi, weight.t()).cuda(0)
            results.append(yi)
            print('rank ', i)
            print('yi ',yi.size())

        y = torch.cat(results, dim=1) + self.bias
        return y

# Example usage:
model = DynamicColumnParallelLinear(1000, 500)
x = torch.randn(32, 1000)
output = model(x)
print(output.shape) # Should be torch.Size([32, 500])
