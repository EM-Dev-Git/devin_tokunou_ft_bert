import os
import urllib.request
import torch

BERT_BASE_UNCASED_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"

def download_bert_files():
    """
    BERTの語彙ファイルと事前学習済みモデルをダウンロードする関数
    
    機能:
    1. BERT-base-uncasedの語彙ファイルをダウンロード
    2. Transformersライブラリを使用して事前学習済みBERTモデルをダウンロードし、保存
    
    引数:
        なし
        
    戻り値:
        なし
        
    副作用:
        - ~/bert_finetuning_project/vocab/bert-base-uncased-vocab.txt ファイルを作成
        - ~/bert_finetuning_project/weights/pytorch_model.bin ファイルを作成
        - ダウンロード状況を標準出力に表示
    """
    
    vocab_dir = os.path.join(os.path.expanduser("~"), "bert_finetuning_project", "vocab")
    weights_dir = os.path.join(os.path.expanduser("~"), "bert_finetuning_project", "weights")
    
    vocab_file_path = os.path.join(vocab_dir, "bert-base-uncased-vocab.txt")
    if not os.path.exists(vocab_file_path):
        print("語彙ファイルをダウンロード中...")
        urllib.request.urlretrieve(BERT_BASE_UNCASED_URL, vocab_file_path)
        print("語彙ファイルをダウンロード完了")
    else:
        print("語彙ファイルは既に存在します")
    
    if not os.path.exists(os.path.join(weights_dir, "pytorch_model.bin")):
        print("事前学習済みモデルをダウンロード中...")
        from transformers import BertModel
        model = BertModel.from_pretrained('bert-base-uncased')
        torch.save(model.state_dict(), os.path.join(weights_dir, "pytorch_model.bin"))
        print("事前学習済みモデルをダウンロード完了")
    else:
        print("事前学習済みモデルは既に存在します")
        
if __name__ == "__main__":
    download_bert_files()
