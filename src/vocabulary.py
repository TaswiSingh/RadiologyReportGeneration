from collections import Counter


class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq

        self.word2idx = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3
        }

        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def tokenize(self, text):
        return text.lower().strip().split()

    def build(self, reports):
        counter = Counter()

        for report in reports:
            counter.update(self.tokenize(report))

        idx = len(self.word2idx)

        for word, freq in counter.items():
            if freq >= self.min_freq:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokens = self.tokenize(text)

        ids = [self.word2idx["<bos>"]]

        for token in tokens:
            ids.append(
                self.word2idx.get(
                    token,
                    self.word2idx["<unk>"]
                )
            )

        ids.append(self.word2idx["<eos>"])

        return ids