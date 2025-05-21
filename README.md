# 職種分類のためのBERTファインチューニング

このプロジェクトは、BERT（Bidirectional Encoder Representations from Transformers）モデルを職種説明文の分類にファインチューニングするための完全なパイプラインを実装しています。このシステムにより、ユーザーは事前学習済みのBERTモデルを取得し、データサイエンティスト、機械学習エンジニア、ソフトウェアエンジニア、コンサルタントなどの役割に求人情報を認識・分類するように適応させることができます。

## プロジェクト概要

このプロジェクトは、4つの主要段階からなる順次パイプラインで構成されています：

1. **データ準備**: 職種説明文とそれに対応するカテゴリラベルのサンプルデータセットを作成
2. **リソース取得**: 事前学習済みBERTモデルと語彙ファイルをダウンロード
3. **モデル学習**: 職種説明文データセットでBERTモデルをファインチューニング
4. **推論**: ファインチューニングされたモデルを使用して新しい職種説明文のカテゴリを予測

## 機能

- テキスト分類のための完全なBERTファインチューニングパイプライン
- 職種説明文を処理するためのカスタムトークナイザー実装
- 学習曲線とモデルパフォーマンス指標の可視化
- サンプルデータ生成機能を備えた事前定義された職種カテゴリ
- 新しい職種説明文を分類するための使いやすい推論モジュール

## プロジェクト構造

```
devin_tokunou_ft_bert/
├── data/                # 学習用・テスト用データセットのディレクトリ
├── vocab/               # BERT語彙ファイルのディレクトリ
├── weights/             # 事前学習済みモデルの重みのディレクトリ
├── results/             # ファインチューニングされたモデルと学習曲線のディレクトリ
├── utils/               # ユーティリティモジュール
│   ├── bert.py          # BERTモデルの実装
│   ├── tokenizer.py     # トークン化機能
│   └── __init__.py      # パッケージ初期化
├── main.py              # メインパイプラインのオーケストレーション
├── download_bert_files.py  # 語彙ファイルと事前学習済み重みのダウンロード
├── create_sample_dataset.py # 学習用・テスト用データセットの生成
├── train_model.py       # ファインチューニングプロセスの実装
└── predict.py           # 推論機能の提供
```

## インストール

1. リポジトリをクローン：
   ```bash
   git clone https://github.com/EM-Dev-Git/devin_tokunou_ft_bert.git
   cd devin_tokunou_ft_bert
   ```

2. 必要な依存関係をインストール：
   ```bash
   pip install torch pandas numpy matplotlib sklearn tqdm transformers
   ```

## 使用方法

### 完全なパイプラインの実行

データ準備から推論までの完全なファインチューニングパイプラインを実行するには：

```bash
python main.py
```

これにより以下が実行されます：
1. BERT語彙ファイルと事前学習済みモデルのダウンロード
2. 職種説明文のサンプルデータセットの作成
3. データセットでのBERTモデルのファインチューニング
4. サンプル職種説明文での推論の実行

### 個別コンポーネント

各コンポーネントを個別に実行することもできます：

1. BERTファイルのダウンロード：
   ```bash
   python download_bert_files.py
   ```

2. サンプルデータセットの作成：
   ```bash
   python create_sample_dataset.py
   ```

3. モデルの学習：
   ```bash
   python train_model.py
   ```

4. 推論の実行：
   ```bash
   python predict.py
   ```

## 語彙ファイルの用途

BERT語彙ファイル（bert-base-uncased-vocab.txt）は、以下のような複数の重要な役割を果たす重要なコンポーネントです：

1. テキスト入力をトークン化するための参照辞書として機能
2. モデル処理のために各トークンに一意のIDを割り当て
3. トークンとその数値表現の間のマッピングを提供
4. 未知の単語を[UNK]トークンに置き換えて処理
5. トークンIDと人間が読める形式のトークン間の変換を可能に

## モデルアーキテクチャ

このプロジェクトは、以下のコンポーネントを持つカスタムBERTアーキテクチャを実装しています：

- **BertTokenizer**: テキストのトークン化とトークンとID間の変換を処理
- **BertModel**: トランスフォーマー層を持つコアBERTアーキテクチャを実装
- **BertForSequenceClassification**: テキスト分類タスク用にBertModelを拡張
- **JobDataset**: 職種説明文データを処理するためのカスタムデータセットクラス

## ファインチューニングの精度向上

モデルの精度を向上させるために、以下のパラメータや設定を調整することができます：

### 学習パラメータの調整 (`train_model.py`)

- **エポック数の増加**: `NUM_EPOCHS`を増やす（現在は5）
  ```python
  NUM_EPOCHS = 10  # 5から10に増加
  ```

- **学習率の調整**: `LEARNING_RATE`を調整する（現在は2e-5）
  ```python
  LEARNING_RATE = 5e-6  # より小さい値に調整
  ```

- **バッチサイズの変更**: `BATCH_SIZE`を変更する（現在は16）
  ```python
  BATCH_SIZE = 32  # より大きい値に調整
  ```

- **シーケンス長の調整**: `MAX_SEQ_LENGTH`を調整する（現在は128）
  ```python
  MAX_SEQ_LENGTH = 256  # より長いシーケンスを処理
  ```

### モデルアーキテクチャの調整 (`utils/bert.py`)

- **ドロップアウト率の調整**: `dropout`率を調整する（現在は0.1）
  ```python
  self.dropout = nn.Dropout(0.2)  # ドロップアウト率を増加
  ```

- **BERTの設定パラメータ調整**: `BertConfig`のパラメータを調整する
  ```python
  config = BertConfig(
      vocab_size=30522,
      hidden_size=768,
      num_hidden_layers=12,  # 層の数を増やす
      num_attention_heads=12,
      intermediate_size=3072,
      hidden_act="gelu",
      hidden_dropout_prob=0.1,
      attention_probs_dropout_prob=0.1,
      max_position_embeddings=512,
      type_vocab_size=2,
      initializer_range=0.02,
      layer_norm_eps=1e-12
  )
  ```

### 最適化手法の変更 (`train_model.py`)

- **オプティマイザの変更**: 現在はAdamを使用
  ```python
  # Adamの代わりにAdamWを使用
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
  ```

- **学習率スケジューラの追加**:
  ```python
  # 学習率スケジューラを追加
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
  ```

### データ拡張 (`create_sample_dataset.py`)

- **トレーニングデータの増加**: より多くのサンプルデータを生成する
- **データ拡張技術の適用**: 同義語置換、ランダムな単語削除などの技術を適用

これらのパラメータを調整することで、モデルの精度を向上させることができます。最適な設定は、データセットやタスクの性質によって異なるため、複数の設定を試して最適な組み合わせを見つけることをお勧めします。

## ライセンス

このプロジェクトは教育および研究目的で利用可能です。
