import pandas as pd

from tokenizer import ReportTokenizer

df = pd.read_csv("data/train.csv")

tokenizer = ReportTokenizer("data/vocab.pkl")

report = df.iloc[0]["report"]

print("Original Report:\n")
print(report)

tokens = tokenizer.encode(report)

print("\nToken IDs:\n")
print(tokens[:30])

decoded = tokenizer.decode(tokens)

print("\nDecoded Report:\n")
print(decoded)