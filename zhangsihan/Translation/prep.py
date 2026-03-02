import json
import numpy as np 
import torch 
from collections import Counter

class TranslationProcessor:
    def __init__(self, train_path, sample_size=50000, min_freq=2):
        self.train_path = train_path
        self.sample_size = sample_size
        self.min_freq = min_freq
        
        print(f"正在读取数据 (目标: {sample_size} 条)...")
        self.zh_data, self.en_data = self._load_raw_data()
        
        print(f"正在构建词表 (当前样本量: {len(self.zh_data)})...")
        self.src_vocab = self._build_vocab(self.en_data, is_chinese=False)
        self.tgt_vocab = self._build_vocab(self.zh_data, is_chinese=True)

        self.reverse_tgt_vocab = {v: k for k, v in self.tgt_vocab.items()}
        
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)

    @property
    def sos_idx(self): return self.tgt_vocab['S']
    
    @property
    def eos_idx(self): return self.tgt_vocab['E']

    def _load_raw_data(self):
        zh_list, en_list = [], []
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.sample_size:
                    break
                try:
                    item = json.loads(line)
                    zh_list.append(item['chinese'])
                    en_list.append(item['english'])
                except Exception as e:
                    continue
        return zh_list, en_list

    def _build_vocab(self, data, is_chinese):
        counter = Counter()
        for sentence in data:

            tokens = list(sentence) if is_chinese else sentence.lower().split()
            counter.update(tokens)
        
        vocab = {'P': 0, 'U': 1, 'S': 2, 'E': 3}
        for key, value in counter.items():
            if value >= self.min_freq:
                if key not in vocab:
                    vocab[key] = len(vocab)
        return vocab

    def encode_src(self, sentence):

        tokens = sentence.lower().split()

        ids = [self.src_vocab.get(w, 1) for w in tokens] 
        return torch.LongTensor([ids]) 

    def decode_tgt(self, indices):

        res = [self.reverse_tgt_vocab.get(idx, '') for idx in indices if idx > 3]
        return "".join(res)

    def make_batch(self, batch_size=32):
        current_data_len = len(self.en_data)
        indices = np.random.choice(current_data_len, min(batch_size, current_data_len))
        
        input_batch, output_batch, target_batch = [], [], []

        for i in indices:
            en_ids = [self.src_vocab.get(w, 1) for w in self.en_data[i].lower().split()]
            zh_in_ids = [self.tgt_vocab['S']] + [self.tgt_vocab.get(c, 1) for c in list(self.zh_data[i])]
            zh_out_ids = [self.tgt_vocab.get(c, 1) for c in list(self.zh_data[i])] + [self.tgt_vocab['E']]
            
            input_batch.append(en_ids)
            output_batch.append(zh_in_ids)
            target_batch.append(zh_out_ids)

        def padding(batch_list):
            max_len = max(len(s) for s in batch_list)
            return [s + [0] * (max_len - len(s)) for s in batch_list]

        return (torch.LongTensor(padding(input_batch)), 
                torch.LongTensor(padding(output_batch)), 
                torch.LongTensor(padding(target_batch)))