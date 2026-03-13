import torch
import pickle
import os
import json
import random
from model import Transformer

PROCESSOR_PATH = r"D:\work\cuiyingMI\projects\project1\SmallProjects\zhangsihan\Translation\professor\processor.pkl"
MODEL_PATH = r"D:\work\cuiyingMI\projects\project1\SmallProjects\zhangsihan\Translation\trainpath\transformer_v1.pth"
VALID_DATA_PATH = r"D:\work\cuiyingMI\projects\project1\SmallProjects\zhangsihan\Translation\dataset\translation2019zh_valid.json"

D_MODEL = 512
N_LAYERS = 4
N_HEADS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_all():
    print(f"正在加载原始词表: {PROCESSOR_PATH}...")
    if not os.path.exists(PROCESSOR_PATH):
        print("错误：找不到 processor.pkl")
        return None, None
        
    with open(PROCESSOR_PATH, 'rb') as f:
        processor = pickle.load(f)

    model = Transformer(
        src_vocab_size=processor.src_vocab_size,
        tgt_vocab_size=processor.tgt_vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=2048,
        n_layers=N_LAYERS,
        n_position=256,
        dropout=0.1 
    ).to(device)

    print(f"正在加载权重: {MODEL_PATH}")
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval() 
        print("模型就绪！")
    except Exception as e:
        print(f"加载失败: {e}")
        return None, None
        
    return model, processor

def translate(model, sentence, processor, device, max_len=50):
    model.eval()
    src_tokens = processor.encode_src(sentence).to(device)
    if (src_tokens == 1).all():
        return "UNK (词汇不在词表中)"

    dec_input = torch.tensor([[processor.sos_idx]], dtype=torch.long).to(device)
    res_tokens = []
    
    for i in range(max_len):
        with torch.no_grad():
            outputs, _, _, _ = model(src_tokens, dec_input)

        last_word_logits = outputs[-1:] 
        next_symbol = last_word_logits.argmax(dim=-1).item()

        if next_symbol == processor.eos_idx:
            break
        if len(res_tokens) > 2 and next_symbol == res_tokens[-1] == res_tokens[-2]:
            break
            
        res_tokens.append(next_symbol)
        next_word_tensor = torch.tensor([[next_symbol]], dtype=torch.long).to(device)
        dec_input = torch.cat([dec_input, next_word_tensor], dim=1)

    return processor.decode_tgt(res_tokens)

def run_validation(model, processor, device, num_samples=20):
    print(f"\n--- 正在从验证集中随机抽取 {num_samples} 条数据进行评估 ---")
    valid_data = []
    with open(VALID_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            valid_data.append(json.loads(line))
    
    samples = random.sample(valid_data, min(num_samples, len(valid_data)))
    
    for i, item in enumerate(samples):
        en_text = item['english']
        gt_zh_text = item['chinese'] 
        
        pred_zh_text = translate(model, en_text, processor, device)
        
        print(f"[{i+1}]")
        print(f"  原文 (EN): {en_text}")
        print(f"  参考 (ZH): {gt_zh_text}")
        print(f"  模型 (ZH): {pred_zh_text}")
        print("-" * 30)

if __name__ == "__main__":
    model, processor = load_all()
    
    if model:

        run_validation(model, processor, device, num_samples=10)

        print("\n评估完成，现在进入交互翻译模式 (输入 q 退出)")
        while True:
            text = input("User (EN) >> ").strip()
            if text.lower() == 'q': break
            if not text: continue
            output = translate(model, text, processor, device)
            print(f"AI (ZH) >> {output}\n")