import torch
from torch.utils.data import Dataset


# Custom PyTorch dataset for handling monster data
# - Loads monster names, descriptions, and visual embeddings
# - Tokenizes descriptions using GPT-2 tokenizer
# - Converts visual embeddings into PyTorch tensors
# - Supports batch processing via DataLoader

class MonsterDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=550):
        self.data = data  # Dictionary with monster names, descriptions and embeddings
        self.tokenizer = tokenizer  # GPT-2 tokenizer for processing text descriptions
        self.max_length = max_length  # Maximum number of tokens for each description

    def __len__(self):
        return len(self.data)  # Return the number of monsters in the dataset

    def __getitem__(self, idx):
        # Retrieve the monster's data at the given index
        monster = self.data[idx]
        description = monster['description']  # Extract the text description of the monster
        embedding = torch.tensor(monster['embedding'],
                                 dtype=torch.float32)  # Convert the visual embedding to a PyTorch tensor

        # Tokenize the description for input into GPT-2
        inputs = self.tokenizer(description, return_tensors='pt', padding='max_length', truncation=True,
                                max_length=self.max_length)

        # Return the visual embedding, input tokens for GPT-2, attention mask, and original description
        return embedding, inputs.input_ids.squeeze(0), inputs.attention_mask.squeeze(0), description
