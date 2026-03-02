import os
import torch 
import pickle
import torch.nn as nn 
import torch.optim as optim 
from tqdm import tqdm
from prep import TranslationProcessor
from model import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True 

TRAIN_DATA_PATH = r'D:\work\cuiyingMI\projects\project1\SmallProjects\zhangsihan\Translation\dataset\translation2019zh_train.json'
SAVE_DIR = r'D:\work\cuiyingMI\projects\project1\SmallProjects\zhangsihan\Translation\trainpath'
PROF_DIR = r"D:\work\cuiyingMI\projects\project1\SmallProjects\zhangsihan\Translation\professor"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PROF_DIR, exist_ok=True) 

SAMPLE_SIZE = 10000 
BATCH_SIZE = 128 
EPOCHS = 300

processor = TranslationProcessor(train_path=TRAIN_DATA_PATH, sample_size=SAMPLE_SIZE)

model = Transformer(
    src_vocab_size=processor.src_vocab_size,
    tgt_vocab_size=processor.tgt_vocab_size,
    d_model=512, 
    n_heads=8, 
    d_ff=2048, 
    n_layers=4, 
    n_position=256,
    dropout=0.1 
).to(device)


criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.98), weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) 
scaler = torch.amp.GradScaler('cuda') 

print("正在生成训练张量...")
all_enc, all_dec, all_tgt = processor.make_batch(SAMPLE_SIZE)
num_batches = SAMPLE_SIZE // BATCH_SIZE

print(f"开始训练，设备: {device}")
pbar = tqdm(range(EPOCHS), desc="训练进度")

for epoch in pbar:
    model.train() 
    total_loss = 0
    perm = torch.randperm(SAMPLE_SIZE)
    
    for i in range(num_batches):
        indices = perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        batch_enc = all_enc[indices].to(device)
        batch_dec = all_dec[indices].to(device)
        batch_tgt = all_tgt[indices].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):

            outputs, _, _, _ = model(batch_enc, batch_dec)
            loss = criterion(outputs, batch_tgt.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    scheduler.step() 
    
    current_lr = optimizer.param_groups[0]['lr']
    pbar.set_postfix({
        "Loss": f"{avg_loss:.4f}", 
        "Epoch": epoch + 1, 
        "LR": f"{current_lr:.6f}"
    })


save_path = os.path.join(SAVE_DIR, 'transformer_v1.pth')
torch.save(model.state_dict(), save_path)
print(f"\n权重已保存: {save_path}")

processor_path = os.path.join(PROF_DIR, "processor.pkl")
with open(processor_path, 'wb') as f:
    pickle.dump(processor, f)
print(f"词表已封存: {processor_path}")