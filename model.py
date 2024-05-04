import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Load CSV
data = pd.read_csv('data0csv')  # Assuming columns 'noisy_text' and 'clean_text'
noisy_texts = data['noisy_text'].tolist()
clean_texts = data['clean_text'].tolist()

# Assuming texts are already pre-processed and tokenized (index-based)
noisy_train, noisy_test, clean_train, clean_test = train_test_split(noisy_texts, clean_texts, test_size=0.2, random_state42)

# Create Tensor datasets
train_dataset = TensorDataset(torch.tensor(noisy_train, dtype=torch.long), torch.tensor(clean_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(noisy_test, dtype=torch.long), torch.tensor(clean_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
import torch.nn as nn

class SequenceAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SequenceAutoencoder, self).__init__()
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, embedding_dim, batch_first=True)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.encoder(x)
        decoded, _ = self.decoder(hidden.unsqueeze(0))
        out = self.output_layer(decoded.squeeze(0))
        return out
def train_autoencoders(train_loader, test_loader, device, num_autoencoders=5):
    autoencoders = []
    for i in range(num_autoencoders):
        autoencoder = SequenceAutoencoder(vocab_size, embedding_dim, hidden_dim)
        if autoencoders:  # Transfer learned weights except for the first one
            autoencoder.load_state_dict(autoencoders[-1].state_dict())
        train_autoencoder(autoencoder, train_loader, device)
        autoencoders.append(autoencoder)
    return autoencoders
def evaluate_autoencoders(autoencoders, test_loader, device):
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            for autoencoder in autoencoders:
                outputs = autoencoder(inputs)
                inputs = outputs.argmax(dim=-1)  # Feed the output as next input
            test_loss += criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1)).item()
 return test_loss
import torch
import torch.nn.functional as F

def binary_tree_search(autoencoders, input_text, target_text, level=0, current_text=None, path=None, best_score=float('inf'), best_path=None):
    if current_text is None:
        current_text = input_text
    if path is None:
        path = []
    if best_path is None:
        best_path = []
        
    if level == len(autoencoders):
        # Calculate similarity index (lower CrossEntropy means higher similarity)
        current_score = F.cross_entropy(current_text, target_text, reduction='mean').item()
        if current_score < best_score:
            return (path.copy(), current_score)
        return (best_path, best_score)
    
    # Case 1: Do not use the current autoencoder
    path_without = path.copy()
    result_without, score_without = binary_tree_search(autoencoders, input_text, target_text, level + 1, current_text, path_without, best_score, best_path)
    
    # Case 2: Use the current autoencoder
    path_with = path.copy()
    path_with.append(level)
    autoencoder_output = autoencoders[level](current_text)
    result_with, score_with = binary_tree_search(autoencoders, input_text, target_text, level + 1, autoencoder_output, path_with, best_score, best_path)
    
    # Determine which path is better
    if score_without < score_with:
        return result_without, score_without
    else:
        return result_with, score_with

