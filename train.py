#!/usr/bin/env python3

import torch
from torchtext import data
from torchtext import datasets

# import argparse
# parser = argparse.ArgumentParser()

EMBEDDING_DIM = 16
PADDING_IDX = 0

# num_embeddings, embedding_dim, padding_idx, input_size, hidden_size, num_layers, dropout, bidirectional):

def main():
  src = data.Field(include_lengths=True, tokenize=list)
  tgt = data.Field(include_lengths=True, tokenize=list)

  mt_train = datasets.TranslationDataset(
    path='data/test', exts=('.src', '.tgt'),
    fields=(src, tgt))
  """mt_dev = datasets.TranslationDataset(
      path='data/dev', exts=('.en', '.ko'),
      fields=(src,tgt))
  """

  print("Building vocabularies..")
  src.build_vocab(mt_train)
  tgt.build_vocab(mt_train)

  print("Making batches..")
  train_iter = data.BucketIterator(
    dataset=mt_train, batch_size=64,
    device=-1,
    # sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg))
    sort_key=lambda x: -len(x.src)
  )

  from spacer import Spacer
  model = Spacer(len(src.vocab), EMBEDDING_DIM, src.vocab.stoi)

  for batch_idx, batch in enumerate(train_iter):
    print(batch_idx)


  print('done')


if __name__ == "__main__":
  main()
