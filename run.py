import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

# --- Configuration ---
LOCAL_MODEL_PATH = './distilbert-local'

DATA_FOLDER = 'twitter-datasets'
TRAIN_POS_FILE = os.path.join(DATA_FOLDER, 'train_pos.txt')
TRAIN_NEG_FILE = os.path.join(DATA_FOLDER, 'train_neg.txt')
TEST_DATA_FILE = os.path.join(DATA_FOLDER, 'test_data.txt')
SUBMISSION_FILE = 'submission.csv'

# Hyperparameters
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 2e-5

# --- Custom Dataset ---
class TweetDataset(Dataset):
    """PyTorch Dataset for loading and tokenizing tweet data."""
    def __init__(self, tweets, labels, tokenizer, max_len):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        output = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        # Labels are optional (not present for test data)
        if self.labels is not None:
            output['labels'] = torch.tensor(self.labels[item], dtype=torch.long)
            
        return output

# --- Model Definition ---
class SentimentClassifier(torch.nn.Module):
    """
    Transformer-based sentiment classifier.
    Uses the [CLS] token's output for classification.
    """
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(LOCAL_MODEL_PATH)
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        output = self.drop(cls_output)
        return self.out(output)

def create_data_loader(tweets, labels, tokenizer, max_len, batch_size, shuffle=False):
    """Creates a DataLoader for the given data."""
    ds = TweetDataset(
        tweets=tweets,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2,
        shuffle=shuffle
    )

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    """Performs one epoch of training."""
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, loss_fn, device):
    """Evaluates the model on a given dataset."""
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# --- NEW FUNCTION TO CREATE SUBMISSION FILE ---
def create_submission_file(model, tokenizer, device):
    """
    Loads the best model, predicts on the test set, and writes the submission file.
    """
    print("\n--- Generating Submission File ---")

    # Load the best performing model state
    model.load_state_dict(torch.load('best_model_state.bin',map_location=torch.device('cpu')))
    model = model.eval() # Set model to evaluation mode
    
    print("Loading and preparing test data...")
    with open(TEST_DATA_FILE, 'r', encoding='utf-8') as f:
        test_tweets_lines = f.readlines()

    # The test file is formatted as "id,tweet text"
    test_ids = [int(line.split(',', 1)[0]) for line in test_tweets_lines]
    test_tweets = [line.split(',', 1)[1] for line in test_tweets_lines]

    # Create a DataLoader for the test set (no labels)
    test_data_loader = create_data_loader(test_tweets, None, tokenizer, MAX_LEN, BATCH_SIZE)

    predictions = []
    with torch.no_grad():
        for d in tqdm(test_data_loader, desc="Predicting on test data"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())

    # Convert predictions (0, 1) to the required format (-1, 1)
    submission_preds = [-1 if p == 0 else 1 for p in predictions]

    print(f"Writing submission to '{SUBMISSION_FILE}'...")
    with open(SUBMISSION_FILE, 'w') as f:
        f.write("Id,Prediction\n")
        for tweet_id, pred in zip(test_ids, submission_preds):
            f.write(f"{tweet_id},{pred}\n")
            
    print("Submission file created successfully!")

def main():
    """Main execution function."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data and Tokenizer from local path ---
    print(f"Loading tokenizer from local path: {LOCAL_MODEL_PATH}...")
    tokenizer = DistilBertTokenizer.from_pretrained(LOCAL_MODEL_PATH)

    print("Loading and preparing tweet data...")
    with open(TRAIN_POS_FILE, 'r', encoding='utf-8') as f:
        pos_tweets = f.readlines()
    with open(TRAIN_NEG_FILE, 'r', encoding='utf-8') as f:
        neg_tweets = f.readlines()

    tweets = [t.strip() for t in pos_tweets + neg_tweets]
    labels = [1] * len(pos_tweets) + [0] * len(neg_tweets)

    train_tweets, val_tweets, train_labels, val_labels = train_test_split(
        tweets, labels, test_size=0.1, random_state=42
    )

    train_data_loader = create_data_loader(train_tweets, train_labels, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=True)
    val_data_loader = create_data_loader(val_tweets, val_labels, tokenizer, MAX_LEN, BATCH_SIZE)
    
    # --- 2. Initialize Model and Optimizer ---
    print("Initializing model from local path...")
    model = SentimentClassifier(n_classes=2).to(device)

    # --- 4. Create Submission File ---
    create_submission_file(model, tokenizer, device)


if __name__ == '__main__':
    main()
