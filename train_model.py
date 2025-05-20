import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from utils.tokenizer import BertTokenizer
from utils.bert import BertConfig, BertForSequenceClassification
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")


BASE_DIR = os.path.join(os.path.expanduser("~"), "bert_finetuning_project")
VOCAB_FILE = os.path.join(BASE_DIR, "vocab", "bert-base-uncased-vocab.txt")
WEIGHTS_FILE = os.path.join(BASE_DIR, "weights", "pytorch_model.bin")
TRAIN_FILE = os.path.join(BASE_DIR, "data", "train.csv")
TEST_FILE = os.path.join(BASE_DIR, "data", "test.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
NUM_LABELS = 4  # 4つの職種カテゴリ

class JobDataset(Dataset):
    """求人情報のデータセット"""
    
    def __init__(self, csv_file, tokenizer, max_length=128):
        """
        初期化
        csv_file: データファイルのパス
        tokenizer: トークナイザー
        max_length: 最大シーケンス長
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        
        tokens = self.tokenizer.tokenize(text)
        
        if len(tokens) > self.max_length - 2:  # [CLS]と[SEP]のための2トークン
            tokens = tokens[:self.max_length - 2]
            
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        padding = [0] * (self.max_length - len(input_ids))
        input_ids += padding
        
        attention_mask = [1] * len(tokens) + [0] * len(padding)
        
        token_type_ids = [0] * self.max_length
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': label
        }

def train_model():
    """BERTモデルの学習を実行"""
    
    tokenizer = BertTokenizer(VOCAB_FILE, do_lower_case=True)
    
    train_dataset = JobDataset(TRAIN_FILE, tokenizer, MAX_SEQ_LENGTH)
    test_dataset = JobDataset(TEST_FILE, tokenizer, MAX_SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    config = BertConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )
    
    model = BertForSequenceClassification(config, num_labels=NUM_LABELS)
    
    if os.path.exists(WEIGHTS_FILE):
        pretrained_weights = torch.load(WEIGHTS_FILE)
        
        model_weights = model.state_dict()
        for name, param in pretrained_weights.items():
            if name in model_weights and 'bert' in name:
                model_weights[name] = param
        
        model.load_state_dict(model_weights, strict=False)
        print("事前学習済みの重みを読み込みました")
    else:
        print("事前学習済みの重みが見つかりません。ランダムな初期化を使用します。")
    
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids, token_type_ids, attention_mask)
            
            loss = criterion(logits, labels)
            
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            
            _, preds = torch.max(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        model.eval()
        test_loss = 0.0
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Test]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(input_ids, token_type_ids, attention_mask)
                
                loss = criterion(logits, labels)
                
                test_loss += loss.item()
                
                _, preds = torch.max(logits, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_acc = accuracy_score(test_labels, test_preds)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        if epoch == NUM_EPOCHS - 1:
            print("\n分類レポート:")
            target_names = ['データサイエンティスト', '機械学習エンジニア', 'ソフトウェアエンジニア', 'コンサルタント']
            print(classification_report(test_labels, test_preds, target_names=target_names))
    
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "finetuned_model.pth"))
    print(f"ファインチューニングしたモデルを保存しました: {os.path.join(RESULTS_DIR, 'finetuned_model.pth')}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train')
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), train_accs, label='Train')
    plt.plot(range(1, NUM_EPOCHS + 1), test_accs, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "learning_curves.png"))
    print(f"学習曲線を保存しました: {os.path.join(RESULTS_DIR, 'learning_curves.png')}")

if __name__ == "__main__":
    train_model()
