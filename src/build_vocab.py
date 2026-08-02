import pickle
import pandas as pd

from vocabulary import Vocabulary

train = pd.read_csv("data/train.csv")

vocab = Vocabulary(min_freq=2)

vocab.build(train["report"].tolist())

with open("data/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("Vocabulary Size:", len(vocab.word2idx))
print("Vocabulary saved to data/vocab.pkl")