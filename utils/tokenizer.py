import collections

class BertTokenizer(object):
    """BERT用のTokenizerクラス。引数に渡された単語列をBERTに入力できる形に変換する"""

    def __init__(self, vocab_file, do_lower_case=True):
        """
        初期化
        vocab_file：ボキャブラリーへのパス
        do_lower_case：前処理で単語を小文字化するかどうか
        """

        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

        self.do_lower_case = do_lower_case

    def load_vocab(self, vocab_file):
        """ボキャブラリーファイルを読み込み、辞書型で返す"""
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
        """入力テキストをトークンに分割"""
        if self.do_lower_case:
            text = text.lower()

        tokens = text.split()

        return tokens

    def convert_tokens_to_ids(self, tokens):
        """トークンリストをIDリストに変換"""
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, self.vocab["[UNK]"]))
        return ids

    def convert_ids_to_tokens(self, ids):
        """IDリストをトークンリストに変換"""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
