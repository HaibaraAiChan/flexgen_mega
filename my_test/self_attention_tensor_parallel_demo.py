import torch
import torch.nn as nn
import torch.multiprocessing as mp

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_splits):
        super(MultiheadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_splits = num_splits
        self.heads_per_split = num_heads // num_splits
        self.head_dim = embed_dim // num_heads
        print('head dim ', self.head_dim)
        self.weight_partitions = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, embed_dim // self.num_splits))
            for _ in range(self.num_splits)
        ])
        self.bias_partitions = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim // self.num_splits))
            for _ in range(self.num_splits)
        ])

    def forward(self, x, split_idx):
        batch_size, seq_len, _ = x.size()
        weight = self.weight_partitions[split_idx]
        print('shape of weight ', weight.size())
        print('x.shape ', x.size())
        bias = self.bias_partitions[split_idx]
        
        q = torch.matmul(x, weight) + bias # Use weight tensor for linear transformation
        print('q shape ', q.size())
        

        k = torch.matmul(x, weight)+ bias  # Use weight tensor for linear transformation
        v = torch.matmul(x, weight)+ bias  # Use weight tensor for linear transformation
        
        q = q.view(batch_size, seq_len, self.heads_per_split, self.head_dim).transpose(1, 2)
        print('after transpose q shape ', q.size())
        k = k.view(batch_size, seq_len, self.heads_per_split, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.heads_per_split, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_scores = attention_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=attention_scores.dtype))

        attention_weights = nn.functional.softmax(scaled_attention_scores, dim=-1)
        print("attention_weights size ", attention_weights.size())
        # a1= torch.matmul(attention_weights, v)
        # print("a1 ", a1.size())
        # a2= a1.transpose(0,1)
        # print("a2 ", a2.size())
        # a3= a2.contiguous()
        # print("a3 ", a3.size())
        # a4= a3.view(batch_size, seq_len, self.head_dim )
        # print("a4 ", a4.size())
        output = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.head_dim)
        return output


def parallel_worker(rank, model, input_data, split_idx, device_id, device_ids, output_tensors):
    print(f"Running process {rank} on device {device_id}")
    
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    input_slice = input_data.to(device)
    
    with torch.no_grad():
        output = model(input_slice, split_idx)
    
    output_tensors.append(output.cpu().detach())
    print(f"Process {rank} completed")

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    embed_dim = 10
    num_heads = 2
    sequence_length = 100
    batch_size = 2
    num_splits = 2  # Number of splits for tensor parallelism
    device_ids = [0, 1]
    # Create a sample input tensor
    input_data = torch.randn(batch_size, sequence_length, embed_dim)

    # Create the multihead self-attention layer
    multihead_attention_layer = MultiheadSelfAttention(embed_dim, num_heads, num_splits)

    # Perform inference
    mp.set_start_method('spawn')
    
    manager = mp.Manager()
    output_tensors = manager.list()
    
    processes = []
    
    for rank, device_id in enumerate(device_ids):
        p = mp.Process(target=parallel_worker, args=(rank, multihead_attention_layer, input_data, rank, device_id, device_ids, output_tensors))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    gathered_output = torch.cat(list(output_tensors), dim=2)  # Combine along dimension 0

    print("Input shape:", input_data.size())
    print("Gathered Output:", gathered_output.size())
