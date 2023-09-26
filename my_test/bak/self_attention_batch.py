import torch
import torch.nn as nn

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Calculate Q, K, and V using matmul transformations
        q = torch.matmul(x, torch.randn(self.embed_dim, self.embed_dim))
        k = torch.matmul(x, torch.randn(self.embed_dim, self.embed_dim))
        v = torch.matmul(x, torch.randn(self.embed_dim, self.embed_dim))

        # Split heads for multihead attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_scores = attention_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=attention_scores.dtype))

        # Apply softmax to get attention weights
        attention_weights = nn.functional.softmax(scaled_attention_scores, dim=-1)

        # Apply attention weights to values and merge heads
        output = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return output

# Example inference
if __name__ == "__main__":
    # Hyperparameters
    embed_dim = 10
    num_heads = 2
    sequence_length = 100
    batch_size = 1

    # Create a sample input tensor
    input_data = torch.randn(batch_size, sequence_length, embed_dim)

    # Create the multihead self-attention layer
    multihead_attention_layer = MultiheadSelfAttention(embed_dim, num_heads)

    # Perform inference
    output = multihead_attention_layer(input_data)

    print("Input shape:", input_data.size())
    print("Output shape:", output.size())
