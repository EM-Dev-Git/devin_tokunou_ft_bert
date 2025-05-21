import os
import torch
from utils.tokenizer import BertTokenizer
from utils.bert import BertConfig, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.join(os.path.expanduser("~"), "bert_finetuning_project")
VOCAB_FILE = os.path.join(BASE_DIR, "vocab", "bert-base-uncased-vocab.txt")
FINETUNED_MODEL_FILE = os.path.join(BASE_DIR, "results", "finetuned_model.pth")
ORIGINAL_MODEL_FILE = os.path.join(BASE_DIR, "weights", "pytorch_model.bin")

MAX_SEQ_LENGTH = 128

CATEGORIES = ['データサイエンティスト', '機械学習エンジニア', 'ソフトウェアエンジニア', 'コンサルタント']

def predict_job_category(text, model_type="finetuned"):
    """
    テキストから職種を予測する関数
    
    機能:
    1. トークナイザーとモデルの初期化
    2. テキストを前処理（トークン化、ID変換、パディングなど）
    3. モデルを使用して職種を予測
    4. 各カテゴリの確率を計算
    
    引数:
        text (str): 職種を予測したい求人情報のテキスト
        model_type (str): 使用するモデルのタイプ。"original"または"finetuned"
        
    戻り値:
        dict: 予測結果を含む辞書
            - predicted_category (str): 予測された職種カテゴリ
            - probabilities (dict): 各カテゴリの確率
            
        モデルファイルが見つからない場合はNoneを返す
    """
    
    tokenizer = BertTokenizer(VOCAB_FILE, do_lower_case=True)
    
    config = BertConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )
    
    model = BertForSequenceClassification(config, num_labels=len(CATEGORIES))
    
    if model_type == "finetuned":
        if os.path.exists(FINETUNED_MODEL_FILE):
            model.load_state_dict(torch.load(FINETUNED_MODEL_FILE, map_location=device))
            print("ファインチューニングされたモデルを読み込みました")
        else:
            print("ファインチューニングされたモデルファイルが見つかりません")
            return None
    elif model_type == "original":
        if os.path.exists(ORIGINAL_MODEL_FILE):
            pretrained_weights = torch.load(ORIGINAL_MODEL_FILE, map_location=device)
            
            model_weights = model.state_dict()
            for name, param in pretrained_weights.items():
                if name in model_weights and 'bert' in name:
                    model_weights[name] = param
            
            model.load_state_dict(model_weights, strict=False)
            print("元のモデル（事前学習済みモデル）を読み込みました")
        else:
            print("元のモデルファイルが見つかりません")
            return None
    else:
        print(f"不明なモデルタイプ: {model_type}")
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
    """
    サンプルテキストに対して元のモデルと学習後のモデルの予測結果を比較する関数
    
    機能:
    1. 各職種カテゴリ（データサイエンティスト、機械学習エンジニア、ソフトウェアエンジニア、コンサルタント）のサンプルテキストを用意
    2. 各サンプルテキストに対して元のモデルと学習後のモデルで職種予測を実行
    3. 両モデルの予測結果と確率を比較表示
    
    引数:
        なし
        
    戻り値:
        なし
        
    副作用:
        予測結果が標準出力に表示される
    """
    
    sample_texts = [
        "Seeking a data scientist with experience in machine learning, statistical analysis, and data visualization. Must have skills in Python, R, and SQL.",
        "Machine learning engineer position to develop and deploy ML models at scale. Experience with TensorFlow, PyTorch, and cloud computing platforms required.",
        "Software engineer needed to develop and maintain web applications. Proficiency in JavaScript, React, and Node.js required.",
        "Management consultant position available. Will advise clients on business strategy, operational improvements, and digital transformation."
    ]
    
    for i, text in enumerate(sample_texts):
        print(f"\n===== サンプル {i+1} =====")
        print(f"テキスト: {text[:100]}...")
        
        original_result = predict_job_category(text, model_type="original")
        
        finetuned_result = predict_job_category(text, model_type="finetuned")
        
        if original_result and finetuned_result:
            print("\n【比較結果】")
            
            print("\n■ 元モデルの予測")
            print(f"予測カテゴリ: {original_result['predicted_category']}")
            print("カテゴリ別確率:")
            for category, prob in original_result['probabilities'].items():
                print(f"  {category}: {prob:.4f}")
            
            print("\n■ 学習後モデルの予測")
            print(f"予測カテゴリ: {finetuned_result['predicted_category']}")
            print("カテゴリ別確率:")
            for category, prob in finetuned_result['probabilities'].items():
                print(f"  {category}: {prob:.4f}")
            
            if original_result['predicted_category'] != finetuned_result['predicted_category']:
                print(f"\n※ 予測カテゴリが変化: {original_result['predicted_category']} → {finetuned_result['predicted_category']}")
            
            print("=" * 50)

if __name__ == "__main__":
    main()
