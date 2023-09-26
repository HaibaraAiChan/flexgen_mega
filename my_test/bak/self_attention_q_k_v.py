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
        self.weight_q_partitions = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, embed_dim // self.num_splits))
            for _ in range(self.num_splits)
        ])
        self.bias_q_partitions = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim // self.num_splits))
            for _ in range(self.num_splits)
        ])
        self.weight_k_partitions = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, embed_dim // self.num_splits))
            for _ in range(self.num_splits)
        ])
        self.bias_k_partitions = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim // self.num_splits))
            for _ in range(self.num_splits)
        ])
        self.weight_v_partitions = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, embed_dim // self.num_splits))
            for _ in range(self.num_splits)
        ])
        self.bias_v_partitions = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim // self.num_splits))
            for _ in range(self.num_splits)
        ])

    def forward(self, x, split_idx):
        batch_size, seq_len, h = x.size()
        weight_q = self.weight_q_partitions[split_idx]
        bias_q = self.bias_q_partitions[split_idx]
        
        q = torch.matmul(x, weight_q) + bias_q

        weight_k = self.weight_k_partitions[split_idx]
        bias_k = self.bias_k_partitions[split_idx]
        
        k = torch.matmul(x, weight_k) + bias_k

        weight_v = self.weight_v_partitions[split_idx]
        bias_v = self.bias_v_partitions[split_idx]
        
        v = torch.matmul(x, weight_v) + bias_v

        # q = q.view(batch_size, seq_len, self.heads_per_split, self.head_dim).transpose(1, 2)
        # k = k.view(batch_size, seq_len, self.heads_per_split, self.head_dim).transpose(1, 2)
        # v = v.view(batch_size, seq_len, self.heads_per_split, self.head_dim).transpose(1, 2)
        
        b = batch_size
        tgt_s= seq_len
        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * self.heads_per_split, tgt_s, self.head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * self.heads_per_split, self.head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * self.heads_per_split, self.head_dim)
        
        k = k_cache.data[:src_s]
        v = v_cache.data[:src_s]
        k[src_s - 1:src_s] = k_new
        v[src_s - 1:src_s] = v_new
        # only update the latest item of K, V in cahce, to save time and energy
        # shape: (b * n_head, head_dim, s)
        k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)
        if k.is_cuda:
            value = self._attention_value(q, k, v, attention_mask.data,
                b, src_s, tgt_s, self.heads_per_split, self.head_dim)
        else:
            q = q.float().cpu()
            k, v = k.float(), v.float()
            value = self._attention_value(q, k, v, attention_mask.data,
                b, src_s, tgt_s, self.heads_per_split, self.head_dim).cuda().half()
            
            
        # attention_scores = torch.matmul(q, k.transpose(-2, -1))
        # scaled_attention_scores = attention_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=attention_scores.dtype))

        # attention_weights = nn.functional.softmax(scaled_attention_scores, dim=-1)

        # output = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.head_dim)
        
        
        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(x.data)

        if donate[0]: x.delete()
        if donate[1]: attention_mask.delete()
        k_new = TorchTensor.create_from_torch(k_new, self)
        v_new = TorchTensor.create_from_torch(v_new, self)

        return TorchTensor.create_from_torch(value, self), k_new, v_new
        
        
        # return output
    
    
    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        
        # shape: (b * n_head, 1, s)
        attn_weights = torch.bmm(q, k)
        # shape: (b, 1, 1, s)
        mask = mask.view(b, 1, 1, src_s)
        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, 1, src_s)
        attn_weights = F.softmax(attn_weights, dim=2)
        
        # shape: (b, n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)
    
    
    
    
    
    
# The rest of your code remains unchanged
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
