import pickle


class ReportTokenizer:

    def __init__(self, vocab_path):
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

    def encode(self, text):
        return self.vocab.numericalize(text)

    def decode(self, ids):
        words = []

        for idx in ids:
            word = self.vocab.idx2word.get(idx, "<unk>")
            words.append(word)

        return " ".join(words)