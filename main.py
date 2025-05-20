import os
import sys

def main():
    """
    プログラムのメイン関数。BERT fine-tuningの全処理を順番に実行する
    
    機能:
    1. 必要なディレクトリの作成
    2. 語彙ファイルと事前学習済みモデルのダウンロード
    3. サンプルデータセットの作成
    4. BERTモデルのファインチューニング
    5. 学習したモデルでの推論実行
    
    引数:
        なし
        
    戻り値:
        なし
        
    副作用:
        - 各モジュールの関数を呼び出すことにより、ファイルの作成、データの生成、モデルの学習、推論の実行などが行われる
        - 処理の進行状況を標準出力に表示
    """
    
    base_dir = os.path.join(os.path.expanduser("~"), "bert_finetuning_project")
    
    for dir_name in ["data", "vocab", "weights", "utils", "results"]:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"ディレクトリを作成しました: {dir_path}")
    
    print("\n1. 語彙ファイルと事前学習済みモデルのダウンロード")
    from download_bert_files import download_bert_files
    download_bert_files()
    
    print("\n2. サンプルデータセットの作成")
    from create_sample_dataset import create_sample_dataset
    create_sample_dataset()
    
    print("\n3. BERTモデルのファインチューニング")
    from train_model import train_model
    train_model()
    
    print("\n4. 学習したモデルで推論を実行")
    from predict import main as predict_main
    predict_main()
    
    print("\n処理が完了しました。")

if __name__ == "__main__":
    main()
