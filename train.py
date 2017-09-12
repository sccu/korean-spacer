#!/usr/bin/env python3
import math

import numpy as np
import torch
from torchtext import data
from torchtext import datasets
import argparse

CUDA_AVAILABLE = torch.cuda.is_available()

EMBEDDING_DIM = 32
HIDDEN_SIZE = 64
NUM_LAYER = 2
DROPOUT = 0.1
BIDIRECTIONAL = True
DEFAULT_BATCH_SIZE = 8


# num_embeddings, embedding_dim, padding_idx, input_size, hidden_size, num_layers, dropout, bidirectional):

def main():
  parser = argparse.ArgumentParser(description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--embedding-size', type=int, dest="embedding_size", help="Embedding size", default=EMBEDDING_DIM)
  parser.add_argument('--hidden-size', type=int, dest="hidden_size", help="Hidden size", default=HIDDEN_SIZE)
  parser.add_argument('--nlayers', type=int, dest="nlayers", help="Number of RNN layers", default=NUM_LAYER)
  parser.add_argument('-b', '--batch-size', type=int, dest="batch_size", help="Batch size", default=DEFAULT_BATCH_SIZE)
  parser.add_argument('--learning-rate', type=float, dest="learning_rate", help="Learning rate", default=0.1)
  parser.add_argument('--batches-per-print', type=int, dest="batches_per_print", help="Number of batches per print",
    default=10)
  parser.add_argument('--prints-per-save', type=int, dest="prints_per_save", help="Number of prints per save",
    default=10)
  parser.add_argument('-m', '--model', help="Path to the model file to load", default=None)
  parser.add_argument('--data', help="train or test", default="train")
  cmd_args = parser.parse_args()

  src = data.Field(include_lengths=True, tokenize=list)
  tgt = data.Field(include_lengths=True, tokenize=list)

  mt_train = datasets.TranslationDataset(
    path='data/%s' % cmd_args.data, exts=('.src', '.tgt'),
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
    dataset=mt_train, batch_size=cmd_args.batch_size,
    device=(None if CUDA_AVAILABLE else -1),
    repeat=False,
    sort_key=lambda x: -len(x.src)
  )

  print("Creating model..")
  from spacer import Spacer
  num_classes = len(tgt.vocab)
  padding_idx = tgt.vocab.stoi["<pad>"]
  model = Spacer(len(src.vocab), num_classes, cmd_args.embedding_size, cmd_args.hidden_size, cmd_args.nlayers, DROPOUT,
    BIDIRECTIONAL, padding_idx=padding_idx)
  if CUDA_AVAILABLE:
    model.cuda(0)

  if cmd_args.model:
    print("Loading model: {}".format(cmd_args.model))
    state_dict = torch.load(cmd_args.model)
    model.load_state_dict(state_dict)

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=cmd_args.learning_rate)

  losses = []
  correct_answer_count = 0
  total_question_count = 0
  for batch_idx, batch in enumerate(train_iter):
    # print("idx: {}, src: {}, tgt: {}".format(batch_idx, batch.src[0].size()[1], batch.trg[0].size()[1]))
    if batch.src[0].size()[0:1] != batch.trg[0].size()[0:1]:
      print("batch.src.size: {}, trg.size: {}".format(batch.src[0].size(), batch.trg[0].size()))
      continue
    assert batch.src[0].size()[0:1] == batch.trg[0].size()[0:1]
    optimizer.zero_grad()

    inputs, src_length = batch.src
    y_ = model(inputs, src_length)
    y_ = y_.view(-1, num_classes)
    y = batch.trg[0]
    y = y.view(-1)
    loss = criterion(y_, y)
    loss.backward()
    optimizer.step()

    losses.append(loss.data[0])
    _, prediction = torch.max(y_, dim=1)
    total_question_count += prediction.size()[0]
    correct_answer_count += (prediction == y).float().sum().data[0]
    if batch_idx % cmd_args.batches_per_print == 0:
      avg_loss = np.mean(losses)

      print("Batch: {}, Loss: {}, Accuracy: {}".format(batch_idx, avg_loss,
        correct_answer_count / total_question_count))

      print("Sentence: {}".format("".join(src.vocab.itos[x[0]] for x in batch.src[0].data)))
      prediction = prediction.view(-1, cmd_args.batch_size)
      print("Prediction: {}".format("".join(tgt.vocab.itos[x[0]] for x in prediction.data)))
      y = y.view(-1, cmd_args.batch_size)
      print("Answer    : {}".format("".join(tgt.vocab.itos[x[0]] for x in y.data)))

      if batch_idx % (cmd_args.batches_per_print * cmd_args.prints_per_save) == 0:
        filename = "models/spacer_{}_{}.pth".format(batch_idx, avg_loss)
        print("Saving a file: {}".format(filename))
        torch.save(model.state_dict(), filename)

      losses = []
      correct_answer_count = 0
      total_question_count = 0

  print('done')


if __name__ == "__main__":
  main()
