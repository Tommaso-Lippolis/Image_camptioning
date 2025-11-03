import torch.nn as nn


# Projection layer to adapt encoder output to GPT-2 input embeddings
# - Expands input dimension with a hidden layer
# - Uses ReLU activation for non-linearity
# - Reduces the final dimension to match GPT-2 embeddings
# - Ensures compatibility between visual embeddings and GPT-2

class ProjectionLayer(nn.Module):
    def __init__(self, embedding_dim, gpt2_embedding_dim, device='cpu'):
        super(ProjectionLayer, self).__init__()
        hidden_dim = gpt2_embedding_dim * 2

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, gpt2_embedding_dim)
        ).to(device)

    def forward(self, x):
        return self.projection(x)
