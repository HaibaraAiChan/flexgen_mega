import torch
import torch.distributed as dist
from torch.nn import Linear
from torch.nn.parallel import DistributedDataParallel as DDP

# Step 1: Initialize the distributed environment
dist.init_process_group(backend='gloo')
world_size = dist.get_world_size()
world_rank = dist.get_rank()

# Step 2: Initialize model parallel groups
def initialize_model_parallel_groups():
    global tensor_model_parallel_group
    if world_rank < world_size // 2:
        tensor_model_parallel_group = dist.new_group(list(range(world_size // 2)))
    else:
        tensor_model_parallel_group = dist.new_group(list(range(world_size // 2, world_size)))

initialize_model_parallel_groups()

def get_tensor_model_parallel_group():
    global tensor_model_parallel_group
    return tensor_model_parallel_group

# Step 3: Define the model, and split it across groups
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = Linear(1000, 500)
        self.fc2 = Linear(500, 100)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Assign different portions of the model to different groups
if world_rank < world_size // 2:
    model = SimpleModel().fc1
    input_shape = (32, 1000)
else:
    model = SimpleModel().fc2
    input_shape = (32, 500)

# Convert the model to DDP
model = DDP(model, process_group=get_tensor_model_parallel_group())

# Step 4: Define a simple training loop
def train(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(10):  # Loop over the dataset multiple times
        # Simulate input data
        inputs = torch.randn(*input_shape)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = torch.mean(outputs)
        loss.backward()
        optimizer.step()
        
        print(f'Rank {world_rank}, Epoch {epoch}, Loss: {loss.item()}')

# Step 5: Start training
train(model)

# Step 6: Cleanup
dist.destroy_process_group()
