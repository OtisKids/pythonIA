import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, heads):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.head_dim = embedding_dim // heads
        
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.fc = nn.Linear(heads * self.head_dim, embedding_dim)
        
    def forward(self, x):
        batch_size, seq_length, embedding_dim = x.size()
        h = self.heads
        
        # Split the embedding into "heads"
        x = x.view(batch_size, seq_length, h, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        
        # Calculate the query, key, and value for each head
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Compute the scaled dot-product attention
        energy = torch.matmul(query, key.permute(0, 1, 3, 2))
        attention = F.softmax(energy / (self.embedding_dim ** 0.5), dim=-1)
        x = torch.matmul(attention, value)
        
        # Concatenate the heads and apply the output layer
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_length, -1)
        x = self.fc(x)
        
        return x

class NeuralNet(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(NeuralNet, self).__init__()
        self.embedding = nn.Embedding(10000, embedding_dim)
        self.attention = SelfAttention(embedding_dim, heads=8)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.attention(x)
        x = F.relu(x)
        x = self.fc(x)
        return x

# Example usage
input_text = "I need a system that can generate invoices and handle payments."
model = NeuralNet(embedding_dim=32, num_classes=10)
tokens = input_text.split()
encoded_input = [hash(token) % 10000 for token in tokens]
encoded_input = torch.tensor(encoded_input).unsqueeze(0)
output = model(encoded_input)
print(output)
