#!/usr/bin/env python3
import numpy as np
import pickle
import os
from scipy.sparse import *
import time
import argparse

# Define file paths
VOCAB_PATH = 'vocab.pkl'
COOC_PATH = 'cooc.pkl'
EMBEDDINGS_PATH = 'glove_embeddings.npy'

def build_cooc_matrix():
    """Build co-occurrence matrix from training data"""
    print("Building co-occurrence matrix...")
    
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    
    data, row, col = [], [], []
    counter = 1
    for fn in ["twitter-datasets/train_pos.txt", "twitter-datasets/train_neg.txt"]:
        with open(fn, encoding='utf-8') as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(f"Processed {counter} lines")
                counter += 1
    
    print("Creating sparse matrix...")
    cooc = coo_matrix((data, (row, col)))
    print("Summing duplicates (this can take a while)...")
    cooc.sum_duplicates()
    
    print(f"Co-occurrence matrix shape: {cooc.shape}, with {cooc.nnz} non-zero entries")
    
    with open(COOC_PATH, "wb") as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
    
    return cooc

def train_glove_embeddings(cooc, embedding_dim=100, epochs=50, learning_rate=0.05, alpha=0.75, x_max=100):
    """Train GloVe embeddings using the co-occurrence matrix"""
    print(f"Training GloVe embeddings with dimension {embedding_dim}...")
    
    # Initialize embeddings
    vocab_size = cooc.shape[0]
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize word vectors and bias terms
    W = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim))
    W_context = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim))
    b_w = np.random.normal(scale=0.1, size=(vocab_size, 1))
    b_c = np.random.normal(scale=0.1, size=(vocab_size, 1))
    
    # Convert to COO format for faster iteration
    if not isinstance(cooc, coo_matrix):
        cooc = cooc.tocoo()
    
    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Shuffle the co-occurrence data
        indices = list(range(len(cooc.data)))
        np.random.shuffle(indices)
        
        # Batch processing for efficiency
        batch_size = 1000000  # Adjust based on your memory constraints
        num_batches = len(indices) // batch_size + (1 if len(indices) % batch_size != 0 else 0)
        
        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]
            
            batch_loss = 0.0
            
            for idx in batch_indices:
                i = cooc.row[idx]
                j = cooc.col[idx]
                X_ij = cooc.data[idx]
                
                # Apply weighting function
                weight = (X_ij / x_max)**alpha if X_ij < x_max else 1
                
                # Compute prediction and error
                prediction = np.dot(W[i], W_context[j]) + b_w[i] + b_c[j]
                error = prediction - np.log(X_ij)
                
                # Compute weighted cost
                cost = weight * (error ** 2)
                batch_loss += cost
                
                # Compute gradients
                grad_w = weight * error * W_context[j]
                grad_c = weight * error * W[i]
                grad_b_w = weight * error
                grad_b_c = weight * error
                
                # Update parameters
                W[i] -= learning_rate * grad_w
                W_context[j] -= learning_rate * grad_c
                b_w[i] -= learning_rate * grad_b_w
                b_c[j] -= learning_rate * grad_b_c
            
            epoch_loss += batch_loss
            
            if batch % 10 == 0:
                # Fix: Convert batch_loss to float before formatting
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch+1}/{num_batches}, Loss: {float(batch_loss)/len(batch_indices):.6f}")
        
        avg_epoch_loss = epoch_loss / len(cooc.data)
        elapsed_time = time.time() - start_time
        # Fix: Convert avg_epoch_loss to float before formatting
        print(f"Epoch {epoch+1}/{epochs} completed in {elapsed_time:.2f}s, Avg Loss: {float(avg_epoch_loss):.6f}")
        
        # Save embeddings every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Final embeddings are the sum of word and context vectors
            embeddings = W + W_context
            np.save(f"{EMBEDDINGS_PATH}.epoch{epoch+1}", embeddings)
    
    # Final embeddings
    embeddings = W + W_context
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"GloVe embeddings saved to {EMBEDDINGS_PATH}")
    
    return embeddings

def main():
    parser = argparse.ArgumentParser(description='Train GloVe embeddings on text data')
    parser.add_argument('--dim', type=int, default=100, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.75, help='Weighting factor alpha')
    parser.add_argument('--xmax', type=int, default=100, help='Maximum co-occurrence count for weighting')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild co-occurrence matrix')
    args = parser.parse_args()
    
    # Build or load co-occurrence matrix
    if not os.path.exists(COOC_PATH) or args.rebuild:
        cooc = build_cooc_matrix()
    else:
        print(f"Loading existing co-occurrence matrix from {COOC_PATH}")
        with open(COOC_PATH, "rb") as f:
            cooc = pickle.load(f)
        print(f"Co-occurrence matrix shape: {cooc.shape}, with {cooc.nnz} non-zero entries")
    
    # Train GloVe embeddings
    embeddings = train_glove_embeddings(
        cooc, 
        embedding_dim=args.dim, 
        epochs=args.epochs, 
        learning_rate=args.lr,
        alpha=args.alpha,
        x_max=args.xmax
    )
    
    print("GloVe training completed!")
    print(f"Embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    main()