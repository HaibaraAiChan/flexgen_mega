import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def parallel_worker(rank, model, input_data, device, output_tensors):
    input_data = input_data.to(device)
    model = model.to(device)
    
    # Split the input tensor
    batch_size = input_data.size(0)
    split_size = batch_size // 2
    start_idx = rank * split_size
    end_idx = (rank + 1) * split_size
    
    input_slice = input_data[start_idx:end_idx]
    
    # Forward pass
    output_slice = model(input_slice)
    output_tensors[device] = output_slice
    print(output_slice)
    return output_slice

if __name__ == "__main__":
    # Create an instance of the model
    model = SimpleModel()
    
    # Define the device IDs for the GPUs
    device_ids = [0, 1]
    
    # Define some dummy input data
    input_data = torch.randn(100, 10)
    
    # Initialize a multiprocessing context
    mp.set_start_method('spawn')
    
    # Create a list to hold the processes
    processes = []
    output_tensors = {}  # To store the output tensors
    for rank, device_id in enumerate(device_ids):
        p = mp.Process(target=parallel_worker, args=(rank, model, input_data, device_id, output_tensors))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()
    sorted(output_tensors.items())
    
    # combined_output = torch.cat(output_tensors.values(), dim=0)
    
    print("Combined Output:", output_tensors)