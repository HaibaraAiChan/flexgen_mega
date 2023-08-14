import torch
import torch.nn as nn
import torch.multiprocessing as mp

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

def parallel_worker(rank, model, input_data, device_id, device_ids, output_tensors):
    print(f"Running process {rank} on device {device_id}")
    
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    
    input_slice = input_data[rank::len(device_ids)]
    input_slice = input_slice.to(device)
    
    with torch.no_grad():  # Add this to prevent gradient tracking
        output = model(input_slice)
    
    output_tensors.append(output.cpu().detach())  # Detach and transfer to CPU
    
    print(f"Process {rank} completed")

if __name__ == "__main__":
    model = SimpleModel()
    # model.fc.weight = torch.load('fc_weights.pth')
    # model.fc.bias = torch.load('fc_bias.pth')
    
    device_ids = [0, 1]
    input_data = torch.randn(100, 10)
    
    mp.set_start_method('spawn')
    
    manager = mp.Manager()
    output_tensors = manager.list()
    
    processes = []
    
    for rank, device_id in enumerate(device_ids):
        p = mp.Process(target=parallel_worker, args=(rank, model, input_data, device_id, device_ids, output_tensors))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    combined_output = torch.cat(list(output_tensors), dim=0)
    print("Combined Output:", combined_output.size())
    # print("Combined Output:", combined_output)
    
    
    torch.save(model.fc.weight, 'fc_weights.pth')
    torch.save(model.fc.bias, 'fc_bias.pth')