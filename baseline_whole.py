import os
import re
import pickle
import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
import random

# --- Configuration ---
DATA_FOLDER = 'twitter-datasets'
VOCAB_MIN_COUNT = 5  # As per cut_vocab.sh which filters counts 1-4
EMBEDDING_DIM = 50
GLOVE_EPOCHS = 25
GLOVE_NMAX = 100
GLOVE_ETA = 0.001
GLOVE_ALPHA = 3 / 4

# --- File Paths ---
TRAIN_POS_FULL_FILE = os.path.join(DATA_FOLDER, 'train_pos_full.txt')
TRAIN_NEG_FULL_FILE = os.path.join(DATA_FOLDER, 'train_neg_full.txt')
TRAIN_POS_FILE = os.path.join(DATA_FOLDER, 'train_pos.txt')
TRAIN_NEG_FILE = os.path.join(DATA_FOLDER, 'train_neg.txt')
TEST_DATA_FILE = os.path.join(DATA_FOLDER, 'test_data.txt')

VOCAB_PKL = 'vocab.pkl'
COOC_PKL = 'cooc.pkl'
EMBEDDINGS_NPY = 'embeddings.npy'
SUBMISSION_CSV = 'submission.csv'


def build_vocab():
    """
    Replicates build_vocab.sh, cut_vocab.sh, and pickle_vocab.py.
    Reads full training data, counts word frequencies, filters by
    min_count, and saves a word-to-index mapping dictionary.
    """
    print("1. Building Vocabulary...")
    if os.path.exists(VOCAB_PKL):
        print("   'vocab.pkl' already exists. Skipping.")
        with open(VOCAB_PKL, "rb") as f:
            return pickle.load(f)

    word_counts = Counter()
    print(f"   Reading {TRAIN_POS_FULL_FILE} and {TRAIN_NEG_FULL_FILE}...")
    for fn in [TRAIN_POS_FULL_FILE, TRAIN_NEG_FULL_FILE]:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                word_counts.update(line.strip().split())

    print(f"   Full vocabulary size: {len(word_counts)}")
    
    # Filter words based on min count (replicates cut_vocab.sh)
    filtered_vocab = {
        word: i
        for i, (word, count) in enumerate(word_counts.items())
        if count >= VOCAB_MIN_COUNT
    }

    print(f"   Cut vocabulary size (count >= {VOCAB_MIN_COUNT}): {len(filtered_vocab)}")

    with open(VOCAB_PKL, 'wb') as f:
        pickle.dump(filtered_vocab, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"   Vocabulary saved to '{VOCAB_PKL}'")
    return filtered_vocab


def build_cooc_matrix(vocab):
    """
    Replicates cooc.py.
    Builds a word-word co-occurrence matrix from the small training set.
    """
    print("\n2. Building Co-occurrence Matrix...")
    if os.path.exists(COOC_PKL):
        print(f"   '{COOC_PKL}' already exists. Skipping.")
        with open(COOC_PKL, "rb") as f:
            return pickle.load(f)

    data, row, col = [], [], []
    counter = 1
    print(f"   Reading {TRAIN_POS_FILE} and {TRAIN_NEG_FILE}...")
    for fn in [TRAIN_POS_FILE, TRAIN_NEG_FILE]:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0] # Filter out-of-vocab words
                for t in tokens:
                    for t2 in tokens:
                        if t != t2: # Don't count self-co-occurrence
                            data.append(1)
                            row.append(t)
                            col.append(t2)

                if counter % 10000 == 0:
                    print(f"   Processed {counter} tweets")
                counter += 1

    cooc = coo_matrix((data, (row, col)), shape=(len(vocab), len(vocab)))
    print("   Summing duplicates (this can take a while)...")
    cooc.sum_duplicates()

    with open(COOC_PKL, 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
        
    print(f"   Co-occurrence matrix saved to '{COOC_PKL}'")
    return cooc


def train_glove_embeddings(cooc):
    """
    Replicates glove_template.py with the SGD update implemented.
    Trains GloVe embeddings based on the co-occurrence matrix.
    """
    print("\n3. Training GloVe Embeddings...")
    if os.path.exists(EMBEDDINGS_NPY):
        print(f"   '{EMBEDDINGS_NPY}' already exists. Skipping.")
        return np.load(EMBEDDINGS_NPY)

    print(f"   {cooc.nnz} nonzero co-occurrence entries")
    print(f"   Using nmax = {GLOVE_NMAX}, cooc.max() = {cooc.max()}")

    print("   Initializing embeddings...")
    vocab_size = cooc.shape[0]
    
    # Initialize embeddings and biases
    xs = np.random.normal(size=(vocab_size, EMBEDDING_DIM))
    ys = np.random.normal(size=(vocab_size, EMBEDDING_DIM))
    xs_biases = np.random.normal(size=(vocab_size))
    ys_biases = np.random.normal(size=(vocab_size))

    # Weighting function
    def weight_func(n):
        if n < GLOVE_NMAX:
            return (n / GLOVE_NMAX) ** GLOVE_ALPHA
        return 1.0

    print(f"   Training for {GLOVE_EPOCHS} epochs...")
    for epoch in range(GLOVE_EPOCHS):
        print(f"   Epoch {epoch + 1}/{GLOVE_EPOCHS}")
        # Shuffle data for each epoch
        zipped = list(zip(cooc.row, cooc.col, cooc.data))
        random.shuffle(zipped)
        
        for ix, jy, n in zipped:
            # Calculate weight
            weight = weight_func(n)
            
            # Calculate difference
            log_n = np.log(1.0 + n) # Use log(1+n) to avoid log(0)
            diff = xs[ix] @ ys[jy] + xs_biases[ix] + ys_biases[jy] - log_n
            
            # Gradient descent updates
            grad_xs = weight * diff * ys[jy]
            grad_ys = weight * diff * xs[ix]
            grad_xs_bias = weight * diff
            grad_ys_bias = weight * diff
            
            xs[ix] -= GLOVE_ETA * grad_xs
            ys[jy] -= GLOVE_ETA * grad_ys
            xs_biases[ix] -= GLOVE_ETA * grad_xs_bias
            ys_biases[jy] -= GLOVE_ETA * grad_ys_bias
            
    # The final embedding is the sum of the two sets of vectors
    final_embeddings = xs + ys
    np.save(EMBEDDINGS_NPY, final_embeddings)
    print(f"   Embeddings saved to '{EMBEDDINGS_NPY}'")
    return final_embeddings


def create_tweet_features(tweets, vocab, embeddings):
    """
    Generates feature vectors for a list of tweets by averaging their
    word embeddings.
    """
    features = np.zeros((len(tweets), EMBEDDING_DIM))
    for i, tweet in enumerate(tweets):
        words = tweet.strip().split()
        word_indices = [vocab.get(word, -1) for word in words]
        
        tweet_embeddings = [
            embeddings[idx] for idx in word_indices if idx != -1
        ]
        
        if tweet_embeddings:
            features[i] = np.mean(tweet_embeddings, axis=0)
            
    return features


def train_classifier(vocab, embeddings):
    """
    Trains a logistic regression classifier on the small training set.
    """
    print("\n4. Training Classifier...")
    
    print(f"   Loading training data from {TRAIN_POS_FILE} and {TRAIN_NEG_FILE}")
    with open(TRAIN_POS_FILE, 'r', encoding='utf-8') as f:
        pos_tweets = f.readlines()
    with open(TRAIN_NEG_FILE, 'r', encoding='utf-8') as f:
        neg_tweets = f.readlines()
        
    all_tweets = pos_tweets + neg_tweets
    # Labels: 1 for positive, -1 for negative
    labels = np.array([1] * len(pos_tweets) + [-1] * len(neg_tweets))
    
    print("   Creating features for training data...")
    train_features = create_tweet_features(all_tweets, vocab, embeddings)

    print("   Training Logistic Regression model...")
    classifier = LogisticRegression(random_state=42, C=0.1, solver='liblinear')
    classifier.fit(train_features, labels)
    
    print("   Training complete.")
    return classifier


def create_submission(classifier, vocab, embeddings):
    """
    Predicts labels for the test set and generates the submission file.
    """
    print("\n5. Creating Submission File...")
    
    print(f"   Loading test data from {TEST_DATA_FILE}")
    with open(TEST_DATA_FILE, 'r', encoding='utf-8') as f:
        test_tweets_lines = f.readlines()

    # The test file is formatted as "id,tweet text"
    test_ids = [int(line.split(',', 1)[0]) for line in test_tweets_lines]
    test_tweets = [line.split(',', 1)[1] for line in test_tweets_lines]

    print("   Creating features for test data...")
    test_features = create_tweet_features(test_tweets, vocab, embeddings)

    print("   Predicting labels...")
    predictions = classifier.predict(test_features)
    
    print(f"   Writing submission to '{SUBMISSION_CSV}'")
    with open(SUBMISSION_CSV, 'w') as f:
        f.write("Id,Prediction\n")
        for tweet_id, pred in zip(test_ids, predictions):
            f.write(f"{tweet_id},{int(pred)}\n")
            
    print("\nSubmission file created successfully!")
    print(f"File '{SUBMISSION_CSV}' is ready.")


def main():
    """Main execution pipeline."""
    # Step 1: Build vocabulary from full dataset
    vocab = build_vocab()
    
    # Step 2: Build co-occurrence matrix from small dataset
    cooc_matrix = build_cooc_matrix(vocab)
    
    # Step 3: Train GloVe word embeddings
    embeddings = train_glove_embeddings(cooc_matrix)
    
    # Step 4: Train a classifier on the embeddings
    model = train_classifier(vocab, embeddings)
    
    # Step 5: Predict on test data and create submission file
    create_submission(model, vocab, embeddings)


if __name__ == '__main__':
    main()