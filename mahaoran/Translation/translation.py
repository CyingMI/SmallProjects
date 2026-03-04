import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import json
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x): # 
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
       
class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super().__init__()
        assert embedding_dim % head == 0
        self.d_k = embedding_dim // head
        self.head = head
        self.linears = nn.modules.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        batch_size = query.size(0)
        query, key, value = [
            model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        return self.linears[-1](x)

    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model) 
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x): # 
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(head=num_heads, embedding_dim=d_model)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) 
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embeddings = Embeddings(vocab=vocab_size, d_model=d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)

    def forward(self, src, mask):
        x = self.embeddings(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(head=num_heads, embedding_dim=d_model, dropout=dropout)
        self.cross_attn = MultiHeadAttention(head=num_heads, embedding_dim=d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embeddings = Embeddings(vocab=vocab_size, d_model=d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)

    def forward(self, tgt, memory, src_mask, tgt_mask):
        x = self.embeddings(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, num_layers=6, d_model=512, num_heads=8, d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, vocab_size=src_vocab_size, max_len=max_len, dropout=dropout)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, vocab_size=tgt_vocab_size, max_len=max_len, dropout=dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)
    def decode(self, tgt, memory, src_mask, tgt_mask):
        return self.decoder(tgt, memory, src_mask, tgt_mask)
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(output)
    
    def generate_mask(self, src, tgt, pad_idx=0):
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len).bool().to(tgt.device))
        tgt_mask = tgt_mask & causal_mask
        return src_mask, tgt_mask

import jieba
import nltk
from nltk.tokenize import word_tokenize
def tokenize_Chinese(text):
    return list(jieba.lcut(text))

def tokenize_English(text):
    return word_tokenize(text)

from torchtext.vocab import build_vocab_from_iterator
def build_vocab(texts):
    vocab = build_vocab_from_iterator(texts)
    return vocab
def read_data(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    src_texts = [item["english"] for item in data]
    tgt_texts = [item["chinese"] for item in data]
    return src_texts, tgt_texts
src_texts, tgt_texts = read_data("./translation2019zh/translation2019zh_train.json")
src_vocab = build_vocab(src_texts)
tgt_vocab = build_vocab(tgt_texts)
def text_to_indices(text, vocab):
    return [vocab.stoi[token] for token in text]

import json
import torch
from torch.utils.data import Dataset, DataLoader
class TranslationDataset(Dataset):
    def __init__(self, data_file, src_vocab, tgt_vocab, max_len=512):
        self.data = []
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"解析错误：{e}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        src_text = self.data[idx]["english"]
        tgt_text = self.data[idx]["chinese"]
        src_tokens = tokenize_English(src_text)
        src_indices = [self.src_vocab.stoi[token] for token in src_tokens]
        tgt_tokens = tokenize_Chinese(tgt_text)
        tgt_indices = [self.tgt_vocab.stoi[token] for token in tgt_tokens]
        src_indices = self.pad_sequence(src_indices, self.max_len)
        tgt_indices = self.pad_sequence(tgt_indices, self.max_len)
        
        return src_indices, tgt_indices
    def pad_sequence(self, seq, max_len):
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq += [self.src_vocab.stoi["<pad>"]] * (max_len - len(seq))
        return seq


from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src, tgt in batch:
        src_batch.append(torch.tensor(src, dtype=torch.long))
        tgt_batch.append(torch.tensor(tgt, dtype=torch.long))
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab.stoi["<pad>"])
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab.stoi["<pad>"])
    return src_batch, tgt_batch
num_layers = 6
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1
batch_size = 4
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
dataset = TranslationDataset("./translation2019zh/translation2019zh_train.json", src_vocab, tgt_vocab, max_len=512)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(src_vocab_size, tgt_vocab_size, num_layers, d_model, num_heads, d_ff, max_len=512, dropout=dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.stoi["<pad>"]) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
pbar = tqdm(dataloader, desc="Training", total=len(dataloader), ncols=100)
for src, tgt in pbar:
    src = src.to(device)
    tgt = tgt.to(device)
    src_mask, tgt_mask = model.generate_mask(src, tgt, pad_idx=src_vocab.stoi["<pad>"])
    output = model(src, tgt, src_mask, tgt_mask)
    loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pbar.set_postfix(loss = loss.item())

model_path = "./model_transformer.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
