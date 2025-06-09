import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from tqdm import tqdm
import numpy as np
import math
import os
from sklearn.model_selection import train_test_split

# --- 全局配置和超参数 ---
# 数据和文件路径
DATA_FOLDER = 'twitter-datasets'
TRAIN_POS_FILE = os.path.join(DATA_FOLDER, 'train_pos.txt')
TRAIN_NEG_FILE = os.path.join(DATA_FOLDER, 'train_neg.txt')

# 模型超参数 (为了演示，这些值都设置得比较小)
VOCAB_SIZE = 10000        # 词汇表大小
MAX_LEN = 64              # 序列最大长度
D_MODEL = 128             # 模型维度 (Embedding ađ Attention
N_HEADS = 4               # 多头注意力的头数
N_LAYERS = 2              # Transformer Encoder的层数
DROPOUT = 0.15             # Dropout比例

# 训练超参数
BATCH_SIZE = 64
EPOCHS = 20               # 从零训练需要更多的Epoch
LEARNING_RATE = 1e-4

# 特殊符号
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
CLS_TOKEN = "<cls>"

# --- 1. 数据处理和词汇表构建 ---

def build_vocab(filepaths, min_freq=2):
    """从文本文件中构建词汇表"""
    word_counts = Counter()
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word_counts.update(line.strip().split())
    
    # 根据最小词频过滤，并限制词汇表大小
    vocab = {word: i + 3 for i, (word, count) in enumerate(word_counts.most_common(VOCAB_SIZE - 3)) if count >= min_freq}
    
    # 添加特殊符号
    vocab[PAD_TOKEN] = 0
    vocab[UNK_TOKEN] = 1
    vocab[CLS_TOKEN] = 2
    
    return vocab

class TweetDataset(Dataset):
    """自定义PyTorch数据集"""
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 在文本前添加[CLS]符号
        tokens = [CLS_TOKEN] + text.split()
        
        # 文本转换为ID
        token_ids = [self.vocab.get(token, self.vocab[UNK_TOKEN]) for token in tokens]
        
        # --- THIS IS THE CORRECTED LINE ---
        # 显式地将 token_ids 转换为 torch.long 类型
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_fn(batch, pad_token_id, max_len):
    """
    自定义的collate_fn，用于处理数据加载器中的批次数据。
    主要功能是进行填充(padding)，使同一批次内的所有序列长度一致。
    """
    token_ids_list, labels_list = zip(*batch)
    
    padded_ids = []
    attention_masks = []

    for ids in token_ids_list:
        # 截断或填充
        if len(ids) > max_len:
            ids = ids[:max_len]
        
        padding_len = max_len - len(ids)
        # 正确行
        padded_id = torch.cat((ids, torch.tensor([pad_token_id] * padding_len, dtype=torch.long)))
        #padded_id = torch.cat((ids, torch.tensor([pad_token_id] * padding_len)))
        attention_mask = torch.tensor([1] * len(ids) + [0] * padding_len)
        
        padded_ids.append(padded_id)
        attention_masks.append(attention_mask)
        
    return {
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels_list)
    }

# --- 2. Transformer 模型组件定义 ---

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # 1. 线性投射，并切分成多头
        # Q, K, V shape: [batch_size, n_heads, seq_len, d_k]
        q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算注意力分数（Scaled Dot-Product Attention）
        # scores shape: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # mask shape: [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 3. 加权求和
        # x shape: [batch_size, n_heads, seq_len, d_k]
        x = torch.matmul(attention, v)
        
        # 4. 拼接多头并进行最终线性变换
        # x shape: [batch_size, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.out_linear(x)

class PositionWiseFeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """单个Transformer Encoder层"""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # 1. 多头注意力 + Add & Norm
        _src = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(_src))
        
        # 2. 前馈网络 + Add & Norm
        _src = self.ffn(src)
        src = self.norm2(src + self.dropout2(_src))
        return src

# --- 3. 完整模型定义 ---

class SentimentTransformer(nn.Module):
    """最终的分类模型"""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, n_classes, dropout, max_len):
        super(SentimentTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 注意：此处的位置编码与论文实现略有不同，采取了更简单的加法方式
        # self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_model * 4, dropout) for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        # 分类头
        self.classifier = nn.Linear(d_model, n_classes)
        self.d_model = d_model

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        seq_len = src.shape[1]
        
        # 1. 词嵌入和位置嵌入
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(src.device)
        pos_emb = self.pos_embedding(pos)
        src = self.dropout(src_emb + pos_emb)
        
        # 2. 通过所有Encoder层
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        # 3. 使用[CLS] token的输出来进行分类
        # [CLS] token在序列的第一个位置
        cls_output = src[:, 0, :]
        
        # 4. 通过分类头得到最终输出
        output = self.classifier(cls_output)
        return output

# --- 4. 训练和评估循环 ---

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    total_loss = 0
    total_correct = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        #try:
        input_ids = batch["input_ids"].to(device)
        #input_ids = torch.tensor(input.ids,dtype=torch.long)
        attention_mask = batch["attention_mask"].to(device).unsqueeze(1).unsqueeze(2) # 适配多头注意力的mask维度
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()
        #except:
        #    print(batch["input_ids"])
        #    print('smth wrong!')
        
    return total_correct / len(data_loader.dataset), total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    total_loss = 0
    total_correct = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).unsqueeze(1).unsqueeze(2)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

    return total_correct / len(data_loader.dataset), total_loss / len(data_loader)

# --- 5. 主执行函数 ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 构建词汇表
    print("Building vocabulary...")
    vocab = build_vocab([TRAIN_POS_FILE, TRAIN_NEG_FILE])
    print(f"Vocabulary size: {len(vocab)}")

    # 2. 加载数据
    print("Loading and preparing data...")
    with open(TRAIN_POS_FILE, 'r', encoding='utf-8') as f:
        pos_tweets = f.readlines()
    with open(TRAIN_NEG_FILE, 'r', encoding='utf-8') as f:
        neg_tweets = f.readlines()
    
    tweets = [t.strip() for t in pos_tweets + neg_tweets]
    labels = [1] * len(pos_tweets) + [0] * len(neg_tweets)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        tweets, labels, test_size=0.1, random_state=42
    )

    # 3. 创建DataLoaders
    # 使用 functools.partial 来包装 collate_fn
    from functools import partial
    collate_with_padding = partial(collate_fn, pad_token_id=vocab[PAD_TOKEN], max_len=MAX_LEN)

    train_dataset = TweetDataset(train_texts, train_labels, vocab, MAX_LEN)
    val_dataset = TweetDataset(val_texts, val_labels, vocab, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_with_padding)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_with_padding)

    # 4. 初始化模型、损失函数和优化器
    model = SentimentTransformer(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        n_classes=2,
        dropout=DROPOUT,
        max_len=MAX_LEN
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # 5. 训练循环
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_acc, val_loss = eval_model(model, val_loader, loss_fn, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

if __name__ == '__main__':
    main()