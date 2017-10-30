import pandas as pd
from utils import preprocess

df = pd.read_csv('./data/train.csv')

with open('./data/fasttext-inputs.txt', 'w') as f:
  for text in df.text:
      f.write(' '.join(preprocess(text).lower().split()) + '\n')

print('Done!')
