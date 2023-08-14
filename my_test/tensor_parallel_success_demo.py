import torch
import torch.nn as nn
import torch.multiprocessing as mp

class SplitLinear(nn.Module):
    def __init__(self, in_features, out_features, num_splits):
        super(SplitLinear, self).__init__()
        self.num_splits = num_splits
        self.weight_partitions = nn.ParameterList([
            nn.Parameter(torch.randn(out_features // num_splits, in_features))
            for _ in range(num_splits)
        ])
        self.bias_partitions = nn.ParameterList([
            nn.Parameter(torch.randn(out_features // num_splits))
            for _ in range(num_splits)
        ])
    
    def forward(self, x, split_idx):
        weight = self.weight_partitions[split_idx]
        print('shape of weight ', weight.size())
        bias = self.bias_partitions[split_idx]
        return torch.matmul(x, weight.t()) + bias

def parallel_worker(rank, model, input_data, split_idx, device_id, device_ids, output_tensors):
    print(f"Running process {rank} on device {device_id}")
    
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    input_slice = input_data.to(device)
    # input_slice = input_data[rank::len(device_ids)]
    # input_slice = input_slice.to(device)
    
    with torch.no_grad():
        output = model(input_slice, split_idx)
    
    output_tensors.append(output.cpu().detach())
    print('shape of current output ', output.cpu().detach().size())
    print(f"Process {rank} completed")

if __name__ == "__main__":
    
    num_splits = 2  # Number of splits for tensor parallelism
    input_data = torch.randn(100, 10)
    output_feat = 6
    model = SplitLinear(10, output_feat, num_splits)
    device_ids = [0, 1]
    mp.set_start_method('spawn')
    
    manager = mp.Manager()
    output_tensors = manager.list()
    
    processes = []
    
    for rank, device_id in enumerate(device_ids):
        p = mp.Process(target=parallel_worker, args=(rank, model, input_data, 0, device_id, device_ids, output_tensors))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    gathered_output = torch.cat(list(output_tensors), dim=1)  # Combine along dimension 0
    print("Gathered Output:", gathered_output.size())
