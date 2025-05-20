import os
import torch
from utils.tokenizer import BertTokenizer
from utils.bert import BertConfig, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.join(os.path.expanduser("~"), "bert_finetuning_project")
VOCAB_FILE = os.path.join(BASE_DIR, "vocab", "bert-base-uncased-vocab.txt")
MODEL_FILE = os.path.join(BASE_DIR, "results", "finetuned_model.pth")

MAX_SEQ_LENGTH = 128

CATEGORIES = ['データサイエンティスト', '機械学習エンジニア', 'ソフトウェアエンジニア', 'コンサルタント']

def predict_job_category(text):
    """テキストから職種を予測"""
    
    tokenizer = BertTokenizer(VOCAB_FILE, do_lower_case=True)
    
    config = BertConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )
    
    model = BertForSequenceClassification(config, num_labels=len(CATEGORIES))
    
    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        print("ファインチューニングされたモデルを読み込みました")
    else:
        print("モデルファイルが見つかりません")
        return None
    
    model = model.to(device)
    
    model.eval()
    
    tokens = tokenizer.tokenize(text)
    
    if len(tokens) > MAX_SEQ_LENGTH - 2:  # [CLS]と[SEP]のための2トークン
        tokens = tokens[:MAX_SEQ_LENGTH - 2]
        
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    padding = [0] * (MAX_SEQ_LENGTH - len(input_ids))
    input_ids += padding
    
    attention_mask = [1] * len(tokens) + [0] * len(padding)
    
    token_type_ids = [0] * MAX_SEQ_LENGTH
    
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(input_ids, token_type_ids, attention_mask)
    
    probs = torch.nn.functional.softmax(logits, dim=1)
    
    _, pred_class = torch.max(probs, dim=1)
    pred_class = pred_class.item()
    
    probs = probs.cpu().numpy()[0]
    
    result = {
        'predicted_category': CATEGORIES[pred_class],
        'probabilities': {CATEGORIES[i]: float(probs[i]) for i in range(len(CATEGORIES))}
    }
    
    return result

def main():
    """メイン関数"""
    
    sample_texts = [
        "Seeking a data scientist with experience in machine learning, statistical analysis, and data visualization. Must have skills in Python, R, and SQL.",
        "Machine learning engineer position to develop and deploy ML models at scale. Experience with TensorFlow, PyTorch, and cloud computing platforms required.",
        "Software engineer needed to develop and maintain web applications. Proficiency in JavaScript, React, and Node.js required.",
        "Management consultant position available. Will advise clients on business strategy, operational improvements, and digital transformation."
    ]
    
    for i, text in enumerate(sample_texts):
        print(f"\nサンプル {i+1}:")
        print(f"テキスト: {text[:100]}...")
        
        result = predict_job_category(text)
        
        if result:
            print(f"予測カテゴリ: {result['predicted_category']}")
            print("カテゴリ別確率:")
            for category, prob in result['probabilities'].items():
                print(f"  {category}: {prob:.4f}")

if __name__ == "__main__":
    main()
