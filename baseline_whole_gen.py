import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from tqdm import tqdm

# --- Configuration ---
DATA_FOLDER = 'twitter-datasets'
EMBEDDING_DIM = 100  # Dimension for Word2Vec vectors
VOCAB_MIN_COUNT = 5  # Ignore words with frequency lower than this
WORKERS = os.cpu_count() # Use all available CPU cores for training

# --- File Paths ---
TRAIN_POS_FULL_FILE = os.path.join(DATA_FOLDER, 'train_pos_full.txt')
TRAIN_NEG_FULL_FILE = os.path.join(DATA_FOLDER, 'train_neg_full.txt')
TRAIN_POS_FILE = os.path.join(DATA_FOLDER, 'train_pos.txt')
TRAIN_NEG_FILE = os.path.join(DATA_FOLDER, 'train_neg.txt')
TEST_DATA_FILE = os.path.join(DATA_FOLDER, 'test_data.txt')

W2V_MODEL_FILE = 'word2vec.model'
SUBMISSION_CSV = 'submission.csv'


class TweetSentenceIterator:
    """An iterator that reads tweet files and yields lists of tokens for Gensim."""
    def __init__(self, *filenames):
        self.filenames = filenames

    def __iter__(self):
        for filename in self.filenames:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line.strip().split()


def train_word2vec_model():
    """Trains a Word2Vec model using Gensim if one doesn't already exist."""
    print("1. Training Word2Vec model with Gensim...")
    if os.path.exists(W2V_MODEL_FILE):
        print(f"   Model file '{W2V_MODEL_FILE}' found. Loading existing model.")
        model = Word2Vec.load(W2V_MODEL_FILE)
        return model

    print(f"   Reading tweets from full dataset for training...")
    sentences = TweetSentenceIterator(TRAIN_POS_FULL_FILE, TRAIN_NEG_FULL_FILE)

    model = Word2Vec(
        sentences=sentences,
        vector_size=EMBEDDING_DIM,
        window=5,
        min_count=VOCAB_MIN_COUNT,
        workers=WORKERS,
        sg=1,  # Use skip-gram model
    )

    model.save(W2V_MODEL_FILE)
    print(f"   Training complete. Model saved to '{W2V_MODEL_FILE}'.")
    return model


def create_tweet_features(tweets, w2v_model, description="Creating Features"):
    """Generates feature vectors for tweets by averaging their word embeddings."""
    features = np.zeros((len(tweets), w2v_model.vector_size))
    
    # Wrap the loop with tqdm for a progress bar
    for i, tweet in enumerate(tqdm(tweets, desc=description)):
        words = tweet.strip().split()
        tweet_embeddings = [
            w2v_model.wv[word] for word in words if word in w2v_model.wv
        ]
        
        if tweet_embeddings:
            features[i] = np.mean(tweet_embeddings, axis=0)
            
    return features


def train_classifier(w2v_model):
    """
    Trains a logistic regression classifier and logs its performance.
    """
    print("\n2. Training and Evaluating Classifier...")
    
    print(f"   Loading training data from {TRAIN_POS_FILE} and {TRAIN_NEG_FILE}")
    with open(TRAIN_POS_FULL_FILE, 'r', encoding='utf-8') as f:
        pos_tweets = f.readlines()
    with open(TRAIN_NEG_FULL_FILE, 'r', encoding='utf-8') as f:
        neg_tweets = f.readlines()
        
    all_tweets = pos_tweets + neg_tweets
    labels = np.array([1] * len(pos_tweets) + [0] * len(neg_tweets)) # Use 1 for pos, 0 for neg for log_loss

    # --- Create features for all data ---
    all_features = create_tweet_features(all_tweets, w2v_model, "Creating All Features")
    
    # --- Split data for validation ---
    X_train, X_val, y_train, y_val = train_test_split(
        all_features, labels, test_size=0.1, random_state=42
    )
    print(f"   Data split into {len(X_train)} training and {len(X_val)} validation samples.")

    # --- Train on the split training set and log performance ---
    print("   Training on 90% of data for logging purposes...")
    classifier = LogisticRegression(random_state=42, C=0.5, solver='liblinear', max_iter=1000)
    classifier.fit(X_train, y_train)

    # --- Training Log ---
    train_pred_proba = classifier.predict_proba(X_train)
    val_pred_proba = classifier.predict_proba(X_val)
    val_pred = classifier.predict(X_val)

    train_loss = log_loss(y_train, train_pred_proba)
    val_loss = log_loss(y_val, val_pred_proba)
    val_acc = accuracy_score(y_val, val_pred)
    
    print("\n" + "="*25)
    print("   TRAINING LOG")
    print(f"   Train Loss       : {train_loss:.4f}")
    print(f"   Validation Loss  : {val_loss:.4f}")
    print(f"   Validation Accuracy: {val_acc:.4f}")
    print("="*25 + "\n")

    # --- Retrain on the FULL dataset for the best final model ---
    print("   Retraining on 100% of data for final submission model...")
    final_classifier = LogisticRegression(random_state=42, C=0.5, solver='liblinear', max_iter=1000)
    final_classifier.fit(all_features, labels) # Train on all data
    
    print("   Training complete.")
    return final_classifier


def create_submission(classifier, w2v_model):
    """Predicts labels for the test set and generates the submission file."""
    print("\n3. Creating Submission File...")
    
    print(f"   Loading test data from {TEST_DATA_FILE}")
    with open(TEST_DATA_FILE, 'r', encoding='utf-8') as f:
        test_tweets_lines = f.readlines()

    test_ids = [int(line.split(',', 1)[0]) for line in test_tweets_lines]
    test_tweets = [line.split(',', 1)[1] for line in test_tweets_lines]

    test_features = create_tweet_features(test_tweets, w2v_model, "Creating Test Features")

    print("   Predicting labels...")
    # Predict and convert back to {-1, 1} format for submission
    predictions_as_0_1 = classifier.predict(test_features)
    predictions_as_neg1_1 = [1 if p == 1 else -1 for p in predictions_as_0_1]
    
    print(f"   Writing submission to '{SUBMISSION_CSV}'")
    with open(SUBMISSION_CSV, 'w') as f:
        f.write("Id,Prediction\n")
        # Use tqdm for writing submission file as well
        for tweet_id, pred in tqdm(zip(test_ids, predictions_as_neg1_1), total=len(test_ids), desc="Writing Submission"):
            f.write(f"{tweet_id},{int(pred)}\n")
            
    print("\nSubmission file created successfully!")
    print(f"File '{SUBMISSION_CSV}' is ready.")


def main():
    """Main execution pipeline."""
    w2v_model = train_word2vec_model()
    final_model = train_classifier(w2v_model)
    create_submission(final_model, w2v_model)


if __name__ == '__main__':
    main()