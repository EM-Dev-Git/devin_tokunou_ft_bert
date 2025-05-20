import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BertLayerNorm(nn.Module):
    """
    BERTで使用されるLayerNormalization層
    
    機能:
    入力テンソルに対して正規化を適用し、学習可能なスケールとシフトパラメータを適用する
    """

    def __init__(self, hidden_size, eps=1e-12):
        """
        BertLayerNormクラスの初期化メソッド
        
        引数:
            hidden_size (int): 隠れ層の次元数
            eps (float, optional): 数値安定性のための小さな値。デフォルトは1e-12
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # スケールパラメータ
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # シフトパラメータ
        self.variance_epsilon = eps  # 数値安定性のための小さな値

    def forward(self, x):
        """
        順伝播計算
        
        引数:
            x (torch.Tensor): 入力テンソル
            
        戻り値:
            torch.Tensor: 正規化された出力テンソル
            
        処理内容:
            1. 平均と分散を計算
            2. 入力を正規化
            3. スケールとシフトパラメータを適用
        """
        u = x.mean(-1, keepdim=True)  # 平均
        s = (x - u).pow(2).mean(-1, keepdim=True)  # 分散
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)  # 正規化
        return self.gamma * x + self.beta  # スケールとシフトを適用

class BertEmbeddings(nn.Module):
    """
    BERTの埋め込み層
    
    機能:
    単語ID、位置情報、セグメント情報から埋め込みベクトルを生成する
    """

    def __init__(self, config):
        """
        BertEmbeddingsクラスの初期化メソッド
        
        引数:
            config (BertConfig): BERTモデルの設定
        """
        super(BertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        """
        順伝播計算
        
        引数:
            input_ids (torch.Tensor): 入力テキストのID列
            token_type_ids (torch.Tensor, optional): トークンタイプID列。Noneの場合は全て0
            
        戻り値:
            torch.Tensor: 埋め込みベクトル
            
        処理内容:
            1. 入力の形状を確認
            2. 位置IDを生成
            3. トークンタイプIDがない場合は0で初期化
            4. 単語埋め込み、位置埋め込み、トークンタイプ埋め込みを計算
            5. 3つの埋め込みを足し合わせる
            6. LayerNormalizationとDropoutを適用
        """
        seq_length = input_ids.size(1)
        device = input_ids.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertLayer(nn.Module):
    """BERTのTransformerブロック"""

    def __init__(self, config):
        super(BertLayer, self).__init__()

        self.attention = BertAttention(config)

        self.intermediate = BertIntermediate(config)

        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertAttention(nn.Module):
    """BERTのMulti-head Attention"""

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertSelfAttention(nn.Module):
    """Multi-head Attentionの関数"""

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class BertSelfOutput(nn.Module):
    """BERTのSelf-Attention層の出力を処理"""

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediate(nn.Module):
    """BERTのFeedforward層"""

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    """Feedforwardの出力を処理"""

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertEncoder(nn.Module):
    """BERTのTransformerを繰り返す"""

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([layer for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertPooler(nn.Module):
    """BERTの[CLS]トークンの特徴量を抽出"""

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertModel(nn.Module):
    """
    BERTの本体モデル
    
    機能:
    テキストの入力から文脈を考慮した特徴量を抽出する
    """

    def __init__(self, config):
        """
        BertModelクラスの初期化メソッド
        
        引数:
            config (BertConfig): BERTモデルの設定
        """
        super(BertModel, self).__init__()
        self.embeddings = BertEmbeddings(config)  # 埋め込み層
        self.encoder = BertEncoder(config)  # エンコーダー層
        self.pooler = BertPooler(config)  # プーリング層
        self.output_all_encoded_layers = True  # 全エンコーダー層の出力を返すかどうか

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        """
        順伝播計算
        
        引数:
            input_ids (torch.Tensor): 入力テキストのID列
            token_type_ids (torch.Tensor, optional): トークンタイプID列。Noneの場合は全て0
            attention_mask (torch.Tensor, optional): アテンションマスク。Noneの場合は全て1
            output_all_encoded_layers (bool, optional): 全エンコーダー層の出力を返すかどうか
            
        戻り値:
            tuple: (encoded_layers, pooled_output)
                - encoded_layers: 全エンコーダー層の出力または最終層の出力
                - pooled_output: [CLS]トークンの特徴量
                
        処理内容:
            1. アテンションマスクとトークンタイプIDの初期化
            2. アテンションマスクの拡張と変換
            3. 埋め込み層の適用
            4. エンコーダー層の適用
            5. プーリング層の適用
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # マスクされた位置に大きな負の値

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

class BertConfig(object):
    """
    BERTモデルの設定クラス
    
    機能:
    BERTモデルのアーキテクチャとハイパーパラメータを定義する
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """
        BertConfigクラスの初期化メソッド
        
        引数:
            vocab_size (int): 語彙サイズ
            hidden_size (int, optional): 隠れ層の次元数。デフォルトは768
            num_hidden_layers (int, optional): Transformerブロックの数。デフォルトは12
            num_attention_heads (int, optional): マルチヘッドアテンションのヘッド数。デフォルトは12
            intermediate_size (int, optional): フィードフォワード層の中間サイズ。デフォルトは3072
            hidden_act (str, optional): 活性化関数。デフォルトは"gelu"
            hidden_dropout_prob (float, optional): 隠れ層のドロップアウト確率。デフォルトは0.1
            attention_probs_dropout_prob (float, optional): アテンション確率のドロップアウト確率。デフォルトは0.1
            max_position_embeddings (int, optional): 最大位置埋め込み数。デフォルトは512
            type_vocab_size (int, optional): トークンタイプ埋め込みの語彙サイズ。デフォルトは2
            initializer_range (float, optional): 初期化の範囲。デフォルトは0.02
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

class BertForSequenceClassification(nn.Module):
    """
    BERTを使った文章分類モデル
    
    機能:
    BERTモデルの出力を使用して文章を分類する
    """

    def __init__(self, config, num_labels=4):
        """
        BertForSequenceClassificationクラスの初期化メソッド
        
        引数:
            config (BertConfig): BERTモデルの設定
            num_labels (int, optional): 分類するラベルの数。デフォルトは4
        """
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel(config)  # BERTモデル
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # ドロップアウト
        self.classifier = nn.Linear(config.hidden_size, num_labels)  # 分類器
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        順伝播計算
        
        引数:
            input_ids (torch.Tensor): 入力テキストのID列
            token_type_ids (torch.Tensor, optional): トークンタイプID列
            attention_mask (torch.Tensor, optional): アテンションマスク
            labels (torch.Tensor, optional): 正解ラベル
            
        戻り値:
            torch.Tensor: 各クラスのロジット
            
        処理内容:
            1. BERTモデルで特徴量を抽出
            2. [CLS]トークンの特徴量を取得
            3. ドロップアウトを適用
            4. 線形分類器でクラス分類
        """
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
