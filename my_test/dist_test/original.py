##### AX+b= Y
import torch
import torch.nn as nn
import torch.distributed as dist

dist.init_process_group(backend='gloo')
world_size = dist.get_world_size()
world_rank = dist.get_rank()

class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weights = nn.Parameter(torch.randn(output_size, input_size).to(world_rank))
        self.bias = nn.Parameter(torch.randn(output_size).to(world_rank))

    def forward(self, x):
        y = torch.matmul(x, self.weights.t())  # Perform linear transformation
        y = y + self.bias  # Add bias
        return y

# Example usage:
model = LinearLayer(1000, 500)
x = torch.randn(32, 1000).to(world_rank)
output = model(x)
print('device of output ', output.device)
print(output.shape)  # Should be torch.Size([32, 500])

# Save weights and bias to files
torch.save(model.weights, 'weights.pth')
torch.save(model.bias, 'bias.pth')
# Save x to a file
torch.save(x, 'x.pth')

# Save output to a file
torch.save(output, 'output.pth')

# Load weights and bias from files
loaded_weights = torch.load('weights.pth')
loaded_bias = torch.load('bias.pth')

# Cleanup
dist.destroy_process_group()
