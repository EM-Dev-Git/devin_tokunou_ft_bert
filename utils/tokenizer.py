import collections

class BertTokenizer(object):
    """
    BERT用のTokenizerクラス。テキストをBERTモデルの入力形式に変換する
    
    機能:
    - テキストをトークン（単語）に分割
    - トークンをIDに変換
    - IDをトークンに逆変換
    """

    def __init__(self, vocab_file, do_lower_case=True):
        """
        BertTokenizerクラスの初期化メソッド
        
        引数:
            vocab_file (str): ボキャブラリーファイルへのパス
            do_lower_case (bool, optional): 前処理で単語を小文字化するかどうか。デフォルトはTrue
            
        副作用:
            - ボキャブラリーファイルを読み込む
            - クラス属性（vocab, ids_to_tokens, do_lower_case）を初期化
        """

        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

        self.do_lower_case = do_lower_case

    def load_vocab(self, vocab_file):
        """
        ボキャブラリーファイルを読み込み、辞書型で返すメソッド
        
        引数:
            vocab_file (str): ボキャブラリーファイルへのパス
            
        戻り値:
            collections.OrderedDict: トークンをキー、IDを値とする辞書
            
        処理内容:
            ボキャブラリーファイルを1行ずつ読み込み、各トークンにIDを割り当てる
        """
        vocab = collections.OrderedDict()
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

    def tokenize(self, text):
        """
        入力テキストをトークン（単語）に分割するメソッド
        
        引数:
            text (str): 分割するテキスト
            
        戻り値:
            list: トークン（単語）のリスト
            
        処理内容:
            1. 必要に応じてテキストを小文字化
            2. 空白で単語に分割
        """
        if self.do_lower_case:
            text = text.lower()

        tokens = text.split()

        return tokens

    def convert_tokens_to_ids(self, tokens):
        """
        トークンリストをIDリストに変換するメソッド
        
        引数:
            tokens (list): トークン（単語）のリスト
            
        戻り値:
            list: トークンに対応するIDのリスト
            
        処理内容:
            各トークンをボキャブラリー内のIDに変換
            ボキャブラリーに存在しないトークンは[UNK]のIDに置き換え
        """
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, self.vocab["[UNK]"]))
        return ids

    def convert_ids_to_tokens(self, ids):
        """
        IDリストをトークンリストに変換するメソッド
        
        引数:
            ids (list): IDのリスト
            
        戻り値:
            list: IDに対応するトークン（単語）のリスト
            
        処理内容:
            各IDをトークンに変換
        """
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
