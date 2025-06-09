import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import os

# Define file paths
VOCAB_PATH = 'c:\\Users\\s1ctx\\Downloads\\project_text_classification\\vocab.pkl'
TRAIN_DATA_PATH = [
    'c:\\Users\\s1ctx\\Downloads\\project_text_classification\\twitter-datasets\\train_neg.txt',
    'c:\\Users\\s1ctx\\Downloads\\project_text_classification\\twitter-datasets\\train_pos.txt'
]

# Hyperparameters
EMBEDDING_DIM = 100 # You might adjust this, often matches GloVe dim
MAX_SEQ_LENGTH = 500 # Maximum length of sequences
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
HIDDEN_DIM = 128 # Dimension for the hidden layer

# Load vocabulary
def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, file_paths, vocab, max_seq_length):
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.data = self._load_data(file_paths)

    def _load_data(self, file_paths):
        data = []
        for file_path in file_paths:
            try:
                # Determine label based on filename
                if 'pos' in file_path.lower():
                    label = 1
                elif 'neg' in file_path.lower():
                    label = 0
                else:
                    print(f"Cannot determine label from filename: {file_path}")
                    continue
                    
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            # Process the entire line as text
                            text = line.strip()
                            tokens = text.split()
                            indices = [self.vocab.get(token, self.vocab.get('<unk>', 0)) for token in tokens]
                            data.append((indices, label))
                        except Exception as e:
                            print(f"Error processing line in {file_path}: {e}")
                            continue
            except FileNotFoundError:
                print(f"Error: File not found at {file_path}. Please check the path.")
                continue
            except Exception as e:
                print(f"An error occurred while reading {file_path}: {e}")
                continue
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        indices, label = self.data[idx]
        if len(indices) < self.max_seq_length:
            padded_indices = indices + [self.vocab.get('<pad>', 0)] * (self.max_seq_length - len(indices))
        else:
            padded_indices = indices[:self.max_seq_length]
        return torch.tensor(padded_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Simple Baseline Model (Feedforward Network)
class BaselineModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(BaselineModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        pooled = torch.mean(embedded, dim=1)
        hidden = self.fc1(pooled)
        activated = self.relu(hidden)
        output = self.fc2(activated)
        return output

# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    accuracy = correct_predictions / total_samples
    return total_loss / len(dataloader), accuracy

# Main execution
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    vocab = load_vocab(VOCAB_PATH)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # We know we have 2 classes (positive and negative) from the dataset design
    output_dim = 2
    print(f"Using {output_dim} output classes (0 for negative, 1 for positive)")

    full_dataset = TextDataset(TRAIN_DATA_PATH, vocab, MAX_SEQ_LENGTH)

    if len(full_dataset) == 0:
        print("Error: No data loaded from training files. Please check file paths and format.")
        exit()

    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    print(f"Total data loaded: {len(full_dataset)}")
    print(f"Training data: {train_size}, Test data: {test_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = BaselineModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    print("Training complete!")

    # Save the model
    torch.save(model.state_dict(), 'baseline_model.pt')
    print("Model saved to baseline_model.pt")