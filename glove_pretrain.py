import os
import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from tqdm import tqdm

# --- Configuration ---
# 选择您要使用的本地Glove文件和对应的维度
GLOVE_FILE = 'glove.6B.50d.txt'
EMBEDDING_DIM = 50
#GLOVE_FILE = 'glove.6B.100d.txt'
#EMBEDDING_DIM = 100

DATA_FOLDER = 'twitter-datasets'

# --- File Paths ---
# 使用完整的训练集来训练分类器
TRAIN_POS_FILE = os.path.join(DATA_FOLDER, 'train_pos_full.txt')
TRAIN_NEG_FILE = os.path.join(DATA_FOLDER, 'train_neg_full.txt')


def load_glove_model(glove_file_path):
    """
    加载本地的GloVe词向量文件。
    为了与后续代码兼容，将其加载到gensim的KeyedVectors对象中。
    """
    print(f"1. Loading GloVe model from local file: {glove_file_path}...")
    if not os.path.exists(glove_file_path):
        print("\n错误：找不到指定的GloVe文件。")
        print("请确保glove.6B.100d.txt（或50d）文件与此脚本位于同一目录，或提供完整路径。")
        return None

    # 创建一个KeyedVectors实例
    word_vectors = KeyedVectors(vector_size=EMBEDDING_DIM)
    
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        vectors = []
        words = []
        for line in tqdm(f, desc="Reading GloVe file"):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            words.append(word)
            vectors.append(vector)
    
    word_vectors.add_vectors(words, vectors)
    print("   GloVe model loaded successfully.")
    return word_vectors


def create_tweet_features(tweets, word_vectors, description="Creating Features"):
    """
    通过平均推文中所有单词的词向量来为其生成特征向量。
    """
    # KeyedVectors对象没有 .vector_size 属性, 所以直接使用EMBEDDING_DIM
    features = np.zeros((len(tweets), EMBEDDING_DIM))
    
    for i, tweet in enumerate(tqdm(tweets, desc=description)):
        words = tweet.strip().split()
        # KeyedVectors对象的接口和 .wv 几乎一样
        tweet_embeddings = [
            word_vectors[word] for word in words if word in word_vectors
        ]
        
        if tweet_embeddings:
            features[i] = np.mean(tweet_embeddings, axis=0)
            
    return features


def train_classifier(word_vectors):
    """
    训练逻辑回归分类器并记录其性能。
    """
    if word_vectors is None:
        return

    print("\n2. Training and Evaluating Classifier...")
    
    print(f"   Loading training data from {TRAIN_POS_FILE} and {TRAIN_NEG_FILE}")
    with open(TRAIN_POS_FILE, 'r', encoding='utf-8') as f:
        pos_tweets = f.readlines()
    with open(TRAIN_NEG_FILE, 'r', encoding='utf-8') as f:
        neg_tweets = f.readlines()
        
    all_tweets = pos_tweets + neg_tweets
    # 使用 1 代表积极, 0 代表消极，以便计算log_loss
    labels = np.array([1] * len(pos_tweets) + [0] * len(neg_tweets))

    # --- 为所有数据创建特征 ---
    all_features = create_tweet_features(all_tweets, word_vectors, "Creating All Features")
    
    # --- 将数据分割为训练集和验证集 ---
    X_train, X_val, y_train, y_val = train_test_split(
        all_features, labels, test_size=0.1, random_state=42
    )
    print(f"   Data split into {len(X_train)} training and {len(X_val)} validation samples.")

    # --- 在分割后的训练集上训练并记录性能 ---
    print("   Training on 90% of data for logging purposes...")
    classifier = LogisticRegression(random_state=42, C=0.5, solver='liblinear', max_iter=1000)
    classifier.fit(X_train, y_train)

    # --- 训练日志 ---
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
    
    print("Evaluation complete.")


def main():
    """主执行流程"""
    # 1. 从本地文件加载GloVe模型
    glove_model = load_glove_model(GLOVE_FILE)
    
    # 2. 训练并评估分类器
    #    train_classifier 函数会打印出所有需要的日志信息
    train_classifier(glove_model)


if __name__ == '__main__':
    main()